#!/usr/bin/env python3
import argparse
import copy
import gzip
import hashlib
import json
import os
import ssl
import sys
import time
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from multiprocessing import Pool, Process, Manager, cpu_count, Queue
from queue import Empty
from glob import glob

json_loads = json.loads

from opensearchpy import OpenSearch, helpers
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

def setup_logging(log_file: str = "wazuh_sync.log", error_log_file: str = "wazuh_sync_errors.log"):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.WARNING)
    error_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logging.getLogger(__name__)

def progress_bar(current, total, prefix='', suffix='', length=50, start_time=None):
    if total > 0:
        percent = current / total
    else:
        percent = 0
    
    filled = int(length * percent)
    bar = '=' * filled + '-' * (length - filled)

    elapsed_time = time.time() - start_time if start_time else 0
    docs_per_sec = current / elapsed_time if elapsed_time > 0 else 0
    
    eta_str = ''
    if docs_per_sec > 0 and current > 0 and total > 0:
        remaining_docs = total - current
        eta_seconds = remaining_docs / docs_per_sec
        eta_str = f" | ETA: {int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"

    print(f'\r{prefix} [{bar}] {percent*100:.1f}% {suffix}{eta_str}', end='', flush=True)

def load_state(state_file: str) -> dict:
    default_state = {'last_processed_file': None, 'last_processed_timestamp': None}
    if not os.path.exists(state_file):
        return default_state
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
            if 'last_processed_file' not in state or 'last_processed_timestamp' not in state:
                return default_state
            return state
    except (json.JSONDecodeError, IOError):
        return default_state

def save_state(state_file: str, last_file: str, last_timestamp: str):
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump({'last_processed_file': last_file, 'last_processed_timestamp': last_timestamp}, f, indent=2)
    except IOError:
        pass

def _get_nested_value(data: Dict, field_path: str) -> Any:
    parts = field_path.split('.')
    value = data
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value

def _set_nested_value(data: Dict, field_path: str, value: Any):
    parts = field_path.split('.')
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value

def _delete_nested_value(data: Dict, field_path: str):
    parts = field_path.split('.')
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            return
        current = current[part]
    if parts[-1] in current:
        del current[parts[-1]]

class FieldAnonymizer:
    def __init__(self, salt: str = "default-salt", field_name: str = "field"):
        self.salt = salt.encode()
        self.field_name = field_name
        self.cache = {}
        self.salt_bytes = f"{field_name}:{salt}".encode()

    def anonymize(self, value: str, format_type: str = "string") -> str:
        if not value or value == "-":
            return value

        value_str = str(value)
        cache_key = (value_str, format_type)

        if cache_key in self.cache:
            return self.cache[cache_key]

        hash_input = value_str.encode() + self.salt_bytes
        hash_value = hashlib.md5(hash_input).hexdigest()

        if format_type == "ip":
            octets = [str((int(hash_value[i:i+2], 16) % 254) + 1) for i in range(0, 8, 2)]
            fake_value = ".".join(octets)
        elif format_type == "number":
            fake_value = str(int(hash_value[:8], 16) % 10000)
        else:
            fake_value = f"anon_{hash_value[:8]}"

        self.cache[cache_key] = fake_value
        return fake_value

def reader_process(log_files: List[str], files_queue: Queue, num_workers: int, last_processed_file: Optional[str], stats_queue: Queue):
    logger = logging.getLogger("Reader")
    start_reading = last_processed_file is None

    for file_path in log_files:
        if not start_reading and file_path == last_processed_file:
            start_reading = True

        if not start_reading:
            continue

        files_queue.put(file_path)

    for _ in range(num_workers):
        files_queue.put(None)

def processor_worker(files_queue: Queue, processed_docs_queue: Queue, config: Dict, stats_queue: Queue):
    worker_name = f"Worker-{os.getpid()}"
    logger = logging.getLogger(worker_name)
    anonymizers = {field: FieldAnonymizer(config['salt'], field) for field in config['anonymize_fields']}

    stats = {
        'processed': 0,
        'filtered_location': 0,
        'filtered_timestamp': 0,
        'parse_errors': 0,
        'missing_id': 0,
        'files_processed': 0
    }

    files_seen_by_this_worker = set()

    while True:
        try:
            file_path = files_queue.get()
            if file_path is None:
                break

            filename = os.path.basename(file_path)
            if filename in files_seen_by_this_worker:
                stats_queue.put(('duplicate_file_warning', filename))
            files_seen_by_this_worker.add(filename)

            try:
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore', compresslevel=6) as f:
                    for line in f:
                        try:
                            log_entry = json_loads(line)
                        except (ValueError, TypeError):
                            stats['parse_errors'] += 1
                            continue

                        try:
                            timestamp = _get_nested_value(log_entry, config['timestamp_field'])
                            if config['last_processed_timestamp'] and timestamp and timestamp <= config['last_processed_timestamp']:
                                stats['filtered_timestamp'] += 1
                                continue

                            location = _get_nested_value(log_entry, 'location')
                            if location != config['log_filter']:
                                stats['filtered_location'] += 1
                                continue

                            full_log = _get_nested_value(log_entry, 'full_log') or ''
                            doc_id = hashlib.sha256(full_log.encode()).hexdigest()

                            doc = {'_source': log_entry, '_id': doc_id, 'file_path': file_path}
                            processed_doc = process_document_logic(doc, config['fields'], config['exclude_fields'], config['anonymize_fields'], anonymizers)

                            processed_docs_queue.put(processed_doc)
                            stats['processed'] += 1

                        except (KeyError, AttributeError):
                            stats['parse_errors'] += 1
                            continue

                stats['files_processed'] += 1

            except Exception as e:
                logger.error(f"Error processing {os.path.basename(file_path)}: {e}")

        except Exception as e:
            logger.error(f"Unexpected worker error: {e}")

    for key, value in stats.items():
        stats_queue.put((worker_name + '_' + key, value))

    processed_docs_queue.put(None)

def process_document_logic(doc: Dict, fields: List[str], exclude_fields: List[str], anonymize_fields: Dict[str, str], anonymizers: Dict[str, FieldAnonymizer]) -> Dict:
    original_source = doc.get('_source', {})
    processed_source = {}

    if fields:
        for field in fields:
            value = _get_nested_value(original_source, field)
            if value is not None:
                _set_nested_value(processed_source, field, value)
    else:
        processed_source = copy.deepcopy(original_source)

    if exclude_fields:
        for field in exclude_fields:
            _delete_nested_value(processed_source, field)

    for field_name, field_type in anonymize_fields.items():
        value = _get_nested_value(processed_source, field_name)
        if value is not None:
            anon_value = anonymizers[field_name].anonymize(value, field_type)
            _set_nested_value(processed_source, field_name, anon_value)
            
    doc['_source'] = processed_source
    return doc

class WazuhLocalSync:
    def __init__(self, config: Dict):
        self.config = config
        optimal = get_optimal_config(config['num_workers'])

        if config.get('auto_optimize', True):
            self.batch_size = config.get('batch_size') or optimal['batch_size']
            self.num_workers = config.get('num_workers') or optimal['num_workers']
            self.queue_size_multiplier = optimal['queue_size_multiplier']
            self.progress_update_interval = optimal['progress_update_interval']
        else:
            self.batch_size = config['batch_size']
            self.num_workers = config['num_workers']
            self.queue_size_multiplier = 10000
            self.progress_update_interval = 2

        self.state_file = config['state_file']
        self.timestamp_field = config['timestamp_field']
        
        def create_custom_ssl_context(ca_file: Optional[str]) -> ssl.SSLContext:
            if ca_file:
                return ssl.create_default_context(cafile=ca_file)
            context = ssl._create_unverified_context()
            context.set_ciphers('DEFAULT@SECLEVEL=1')
            return context

        es_config = {'hosts': [config['target_host']]}
        if config['target_user'] and config['target_pass']:
            es_config['http_auth'] = (config['target_user'], config['target_pass'])
        
        if config['target_host'].lower().startswith('https'):
            es_config['ssl_context'] = create_custom_ssl_context(config['target_ca_certs'])
            
        self.es = OpenSearch(**es_config)
        self.stats = {
            'inserted': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'total_docs_found': 0,
            'indices_used': {}
        }

    def _get_index_from_timestamp(self, target_index: str, timestamp: str) -> str:
        if '{date}' not in target_index:
            return target_index

        if not timestamp:
            date_str = datetime.now().strftime('%Y.%m.%d')
        else:
            try:
                timestamp_str = str(timestamp)
                if 'T' in timestamp_str:
                    date_part = timestamp_str.split('T')[0]
                elif ' ' in timestamp_str:
                    date_part = timestamp_str.split(' ')[0]
                else:
                    date_part = timestamp_str[:10]

                date_str = date_part.replace('-', '.')
            except Exception:
                date_str = datetime.now().strftime('%Y.%m.%d')

        return target_index.replace('{date}', date_str)

    def authenticate(self) -> bool:
        try:
            self.es.info()
            return True
        except Exception:
            return False

    def bulk_insert(self, documents: List[Dict], target_index: str) -> Dict:
        logger = logging.getLogger("BulkInsert")
        if not documents:
            return {'success': 0, 'failed': 0}

        def create_actions():
            for doc in documents:
                source = doc.get('_source', {})
                source.setdefault('metadata', {}).update({
                    'processed_at': datetime.now().isoformat(),
                    'original_file': os.path.basename(doc.get('file_path', '')),
                    'anonymized_fields': list(self.config['anonymize_fields'].keys())
                })

                timestamp = source.get(self.timestamp_field)
                index_name = self._get_index_from_timestamp(target_index, timestamp)

                yield {
                    '_op_type': 'index',
                    '_index': index_name,
                    '_id': doc.get('_id'),
                    '_source': source
                }

        success, failed = 0, 0
        
        try:
            for ok, _ in helpers.streaming_bulk(self.es, create_actions(), chunk_size=self.batch_size, raise_on_error=False):
                if ok:
                    success += 1
                else:
                    failed += 1
        except Exception as e:
            logger.error(f"Critical bulk insert error: {e}")
            return {'success': 0, 'failed': len(documents)}

        return {'success': success, 'failed': failed}

    def sync(self):
        logger = logging.getLogger("Sync")
        self.stats['start_time'] = time.time()
        print(f"WAZUH SYNC -> {self.config['target_index']}")

        if not self.authenticate():
            print("Authentication failed")
            return False

        state = load_state(self.state_file)
        self.config['last_processed_timestamp'] = state.get('last_processed_timestamp')
        last_processed_file = state.get('last_processed_file')

        if self.config['last_processed_timestamp']:
            print(f"Resuming from {os.path.basename(last_processed_file or '')} @ {self.config['last_processed_timestamp']}")

        search_pattern = os.path.join(self.config['log_path'], '**', '*.gz')
        all_files = sorted(glob(search_pattern, recursive=True))
        log_files = sorted(all_files)

        if not log_files:
            print("No logs found.")
            return True

        with Manager() as manager:
            queue_size = int(len(log_files) * self.queue_size_multiplier)
            files_queue = manager.Queue(maxsize=len(log_files))
            processed_docs_queue = manager.Queue(maxsize=queue_size)
            stats_queue = manager.Queue()

            worker_args = (files_queue, processed_docs_queue, self.config, stats_queue)
            with Pool(self.num_workers) as pool:
                pool.starmap_async(processor_worker, [worker_args for _ in range(self.num_workers)])

                reader = Process(
                    target=reader_process,
                    args=(log_files, files_queue, self.num_workers, last_processed_file, stats_queue)
                )
                reader.start()

                batch = []
                current_file_in_batch = None
                max_timestamp_in_batch = self.config['last_processed_timestamp']
                file_in_batch = last_processed_file
                workers_done = 0
                last_progress_time = time.time()

                while workers_done < self.num_workers:
                    try:
                        doc = processed_docs_queue.get(timeout=3)

                        if doc is None:
                            workers_done += 1
                            continue

                        self.stats['total_docs_found'] += 1

                        doc_file = doc['file_path']
                        current_file_in_batch = doc_file
                        batch.append(doc)

                        ts = _get_nested_value(doc['_source'], self.timestamp_field)
                        if ts and (max_timestamp_in_batch is None or ts > max_timestamp_in_batch):
                            max_timestamp_in_batch = ts
                            file_in_batch = doc_file

                        if len(batch) >= self.batch_size:
                            result = self.bulk_insert(batch, self.config['target_index'])
                            self.stats['inserted'] += result['success']
                            self.stats['skipped'] += result['failed']
                            if result['success'] > 0 and max_timestamp_in_batch:
                                save_state(self.state_file, file_in_batch, max_timestamp_in_batch)
                            batch.clear()
                            current_file_in_batch = None
                            max_timestamp_in_batch = self.config['last_processed_timestamp']

                        now = time.time()
                        if now - last_progress_time >= self.progress_update_interval:
                            progress_bar(
                                self.stats['inserted'] + self.stats['skipped'], 0,
                                prefix='Processing',
                                suffix=f" Inserted: {self.stats['inserted']:,} | Total: {self.stats['total_docs_found']:,}",
                                start_time=self.stats['start_time']
                            )
                            last_progress_time = now

                    except Empty:
                        if not reader.is_alive() and files_queue.empty():
                            time.sleep(0.05)
                        if workers_done == self.num_workers:
                            break
                        continue

                if batch:
                    result = self.bulk_insert(batch, self.config['target_index'])
                    self.stats['inserted'] += result['success']
                    self.stats['skipped'] += result['failed']
                    if result['success'] > 0 and max_timestamp_in_batch:
                        save_state(self.state_file, file_in_batch, max_timestamp_in_batch)

            reader.join()

            worker_stats = {
                'worker_processed': 0,
                'worker_filtered_location': 0,
                'worker_filtered_timestamp': 0,
                'worker_parse_errors': 0,
                'worker_missing_id': 0,
                'worker_files_processed': 0
            }

            while not stats_queue.empty():
                try:
                    key, value = stats_queue.get_nowait()
                    if key.startswith('Worker-'):
                        metric_type = key.split('_', 1)[1]
                        if metric_type in worker_stats:
                            worker_stats[metric_type] += value
                except Empty:
                    break

            self.stats.update(worker_stats)

        print()
        try:
            self.es.indices.refresh(index=self.config['target_index'])
        except Exception:
            pass

        try:
            response = self.es.count(index=self.config['target_index'])
            docs_in_index = response.get('count', 0)
            self.stats['docs_in_index'] = docs_in_index
        except Exception:
            pass

        self.stats['end_time'] = time.time()
        self.print_stats()

        return True

    def print_stats(self):
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"  Files processed: {self.stats.get('worker_files_processed', 0):,}")
        print(f"  Docs inserted: {self.stats['inserted']:,}")
        print(f"  Docs failed: {self.stats['skipped']:,}")
        print(f"  Filtered (loc): {self.stats.get('worker_filtered_location', 0):,}")
        print(f"  Filtered (ts): {self.stats.get('worker_filtered_timestamp', 0):,}")
        print(f"  Parse Errors: {self.stats.get('worker_parse_errors', 0):,}")
        print(f"  Duration: {duration:.2f}s")
        print("="*50)

def load_config(config_file: str) -> Dict:
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def get_optimal_config(num_cpus: int) -> Dict:
    if num_cpus >= 32:
        return {'num_workers': num_cpus, 'batch_size': 10000, 'queue_size_multiplier': 30000, 'progress_update_interval': 10}
    elif num_cpus >= 16:
        return {'num_workers': num_cpus, 'batch_size': 10000, 'queue_size_multiplier': 25000, 'progress_update_interval': 10}
    elif num_cpus >= 8:
        return {'num_workers': num_cpus, 'batch_size': 5000, 'queue_size_multiplier': 15000, 'progress_update_interval': 5}
    else:
        return {'num_workers': num_cpus, 'batch_size': 1000, 'queue_size_multiplier': 5000, 'progress_update_interval': 2}

def main():
    logger = setup_logging("wazuh_sync.log")
    
    parser = argparse.ArgumentParser(description='Wazuh Local Data Sync')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    args = parser.parse_args()

    config = load_config(args.config)
    base_config_name = os.path.splitext(os.path.basename(args.config))[0]
    state_file_name = f"{base_config_name}_state.json"

    def get_value(config_path, default=None):
        base = config
        for key in config_path.split('.'):
            base = base.get(key) if isinstance(base, dict) else None
        return base if base is not None else default

    num_cpus = max(1, cpu_count() // 2) if cpu_count() > 8 else cpu_count()

    cfg = {
        'target_host': get_value('target.host', 'http://localhost:9200'),
        'target_user': get_value('target.user'),
        'target_pass': get_value('target.password'),
        'target_ca_certs': get_value('target.ca_certs'),
        'target_index': get_value('indices.target_index', 'wazuh-processed'),
        'log_path': get_value('local_logs.path'),
        'log_filter': get_value('local_logs.filter_value', '/logs/nginx/api.production.access.log'),
        'salt': get_value('processing.salt', 'default-salt-please-change'),
        'batch_size': get_value('processing.batch_size'),
        'num_workers': get_value('processing.num_workers', num_cpus),
        'timestamp_field': get_value('processing.timestamp_field'),
        'state_file': state_file_name,
        'fields': get_value('processing.fields'),
        'exclude_fields': get_value('processing.exclude_fields'),
        'anonymize_fields': get_value('processing.anonymize_fields', {}),
    }

    if not cfg['log_path'] or not cfg['timestamp_field']:
        sys.exit("Error: 'log_path' and 'timestamp_field' are required.")
    if not os.path.isdir(cfg['log_path']):
        sys.exit(f"Error: Invalid log path: {cfg['log_path']}")

    syncer = WazuhLocalSync(cfg)
    success = syncer.sync()

    if success:
        logger.info("Sync completed.")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
