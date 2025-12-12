#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
import multiprocessing
import time
import traceback
import gc

sys.path.insert(0, str(Path(__file__).parent))

from core import ConfigLoader, OpenSearchClient, WazuhLogger
from data import LogFetcher, DataPreprocessor
from metrics import L1MetricsCalculator, L2MetricsCalculator, MetricsStorage

def process_chunk(chunk_start, chunk_end, config):
    """
    Worker function to process a time chunk of data.
    """
    pid = os.getpid()
    print(f"[PID {pid}] Starting chunk: {chunk_start} to {chunk_end}")
    
    try:
        # Re-initialize components for this process
        # We create a new client instance per process to avoid connection pooling issues
        # OpenSearchClient singleton reset might be needed if it was already initialized in parent
        # But since we pass config, we can just create a raw instance or handle it carefully.
        # Ideally, OpenSearchClient.get_instance should handle this, or we force new connection.
        
        # Hack: Since OpenSearchClient is a singleton, we need to be careful.
        # In multiprocessing, memory is copied-on-write (Linux) or separate (Windows).
        # On Windows, it's a fresh process, so singleton is None.
        
        client = OpenSearchClient.get_instance(config)
        
        fetcher = LogFetcher(client, config)
        preprocessor = DataPreprocessor(config)
        l1_calc = L1MetricsCalculator(config)
        l2_calc = L2MetricsCalculator(config, client)
        storage = MetricsStorage(client, config)

        # Fetch data with lookback (60 min) to ensure first window is valid
        lookback_time = chunk_start - timedelta(minutes=60)
        fetch_end = chunk_end

        # print(f"[PID {pid}] Fetching logs from {lookback_time} to {fetch_end}")
        all_data = fetcher.fetch_logs(lookback_time, fetch_end)
        all_data = preprocessor.prepare(all_data)

        if all_data.empty:
            print(f"[PID {pid}] No data for {chunk_start}")
            return {'metrics_count': 0, 'l1_count': 0, 'l2_count': 0, 'raw_logs': 0, 'l1_sample_sum': 0, 'l2_user_sample_sum': 0, 'l2_route_sample_sum': 0}

        active_entities = preprocessor.get_active_entities(all_data)
        # print(f"[PID {pid}] Found {len(active_entities)} entities")

        metrics_buffer = []
        metrics_count = 0
        
        # Validation Counters
        stats = {
            'metrics_count': 0,
            'l1_count': 0,
            'l2_count': 0,
            'raw_logs': len(all_data),
            'l1_sample_sum': 0,
            'l2_user_sample_sum': 0,
            'l2_route_sample_sum': 0
        }

        BULK_SIZE = 5000

        # We iterate by entity first for efficiency (filtering dataframe once per entity)
        for idx, entity_id in enumerate(active_entities):
            entity_df = preprocessor.filter_by_entity(all_data, entity_id)
            if entity_df.empty:
                continue
                
            # Iterate STRICTLY by configured window sizes aligned to the hour
            # (e.g. 10m window -> 00:00, 00:10, 00:20...)

            active_windows = config.get('observation_windows', {}).get('enabled', [10, 30, 60])

            for window in active_windows:
                freq = f"{window}min"
                # Generate aligned timestamps for this window size
                # Using freq ensures we get 00:10, 00:20 etc if starting from 00:00
                # chunk_start is aligned to hour boundaries.
                # inclusive='left' avoids processing the end timestamp,
                # which belongs to the next chunk.
                aligned_times = pd.date_range(start=chunk_start, end=chunk_end, freq=freq, inclusive='left')
                
                for current_time in aligned_times:
                    current_time = current_time.to_pydatetime()
                    
                    # Skip logic (kept for future reference, currently pass)
                    pass 

                    window_start = current_time - timedelta(minutes=window)
                    
                    # Optimization: Slice dataframe by time using searchsorted or boolean indexing
                    # Boolean indexing is cleaner but slower. 
                    # Let's trust pandas for now.
                    
                    # Fast slice:
                    # entity_df is already sorted by timestamp? 
                    # LogFetcher usually sorts. If not, we should sort.
                    
                    window_df = entity_df[(entity_df['@timestamp'] >= window_start) & 
                                          (entity_df['@timestamp'] < current_time)]
                    
                    if window_df.empty:
                        # Even if empty, we might want to record zero metrics?
                        continue
                        
                    # L1 Calculation
                    try:
                        l1_metrics = l1_calc.calculate(window_df, window)

                        # Add metadata for bulk save
                        start_time = current_time - timedelta(minutes=window)
                        doc = {
                            '@timestamp': current_time.isoformat(),
                            'window_start_time': start_time.isoformat(),
                            'window_end_time': current_time.isoformat(),
                            'entity_id': entity_id,
                            'observation_window': window,
                            'layer': 'L1',
                            'metric_type': 'entity_metric',
                            'sample_count': l1_metrics.get('total_requests', 0),
                            'metrics': l1_metrics
                        }
                        metrics_buffer.append(doc)
                        stats['l1_count'] += 1
                        stats['l1_sample_sum'] += doc['sample_count']
                    except Exception as e:
                        print(f"[PID {pid}] ERROR in L1 calculation for entity {entity_id}, window {window}min at {current_time}: {str(e)}")
                        pass

                    # L2 Calculation
                    l2_enabled = config.get('metrics', {}).get('layers', {}).get('L2', {}).get('enabled', False)
                    if l2_enabled:
                        try:
                            for dimension in l2_calc.dimensions:
                                dim_results = l2_calc.calculate(window_df, dimension)
                                for result in dim_results:
                                    start_time = current_time - timedelta(minutes=window)
                                    doc = {
                                        '@timestamp': current_time.isoformat(),
                                        'window_start_time': start_time.isoformat(),
                                        'window_end_time': current_time.isoformat(),
                                        'entity_id': entity_id,
                                        'observation_window': window,
                                        'layer': 'L2',
                                        'metric_type': 'user_metric' if result['dimension'] == 'user' else 'l2_metric',
                                        'dimension': result['dimension'],
                                        'dimension_value': result['dimension_value'],
                                        'sample_count': result['sample_count'],
                                        'metrics': result['metrics']
                                    }
                                    if result['dimension'] == 'user':
                                        doc['operator_id'] = result['dimension_value']
                                        stats['l2_user_sample_sum'] += doc['sample_count']
                                    elif result['dimension'] == 'route':
                                        stats['l2_route_sample_sum'] += doc['sample_count']

                                    metrics_buffer.append(doc)
                                    stats['l2_count'] += 1
                                    # print(f"DEBUG: Generated L2 doc: {doc['dimension']}={doc['dimension_value']}")
                        except Exception as e:
                            print(f"[PID {pid}] ERROR in L2 calculation for entity {entity_id}, window {window}min at {current_time}: {str(e)}")
                            traceback.print_exc()
                            pass

                    # Flush buffer if full
                    if len(metrics_buffer) >= BULK_SIZE:
                        storage.save_metrics_bulk(metrics_buffer)
                        stats['metrics_count'] += len(metrics_buffer)
                        metrics_buffer = []
                        gc.collect()  # Force garbage collection after bulk save

            # Free entity DataFrame after processing
            del entity_df

            # Periodic GC every 10 entities
            if (idx + 1) % 10 == 0:
                gc.collect()
                print(f"[PID {pid}] Processed {idx + 1}/{len(active_entities)} entities. Memory cleanup done.")

        # Final flush
        if metrics_buffer:
            storage.save_metrics_bulk(metrics_buffer)
            stats['metrics_count'] += len(metrics_buffer)

        # Free main DataFrame and force final GC
        del all_data
        del metrics_buffer
        gc.collect()

        print(f"[PID {pid}] Finished {chunk_start}. Saved {stats['metrics_count']} metrics (L1: {stats['l1_count']}, L2: {stats['l2_count']}).")
        return stats

    except Exception as e:
        print(f"[PID {pid}] CRITICAL ERROR processing {chunk_start}: {str(e)}")
        traceback.print_exc()
        return {'metrics_count': 0, 'l1_count': 0, 'l2_count': 0, 'raw_logs': 0, 'l1_sample_sum': 0, 'l2_user_sample_sum': 0, 'l2_route_sample_sum': 0}


def main():
    load_dotenv()
    
    # Enable multiprocessing support for Windows
    multiprocessing.freeze_support()

    try:
        config = ConfigLoader().get_all()
        warmup_config = config.get('warmup', {})
        
        if not warmup_config.get('enabled', False):
            print("Warmup disabled in config.")
            return

        start_date = datetime.fromisoformat(warmup_config['start_date'].replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(warmup_config['end_date'].replace('Z', '+00:00'))
        
        print(f"WARMUP STARTED: {start_date} to {end_date}")
        print("Generating dense metrics with parallel processing...")

        # Split into 6-hour chunks to avoid circuit breaker
        chunk_hours = warmup_config.get('chunk_hours', 6)
        chunks = []
        curr = start_date
        while curr < end_date:
            next_chunk = min(curr + timedelta(hours=chunk_hours), end_date)
            chunks.append((curr, next_chunk))
            curr = next_chunk

            if curr >= end_date:
                break

        print(f"Processing {len(chunks)} chunks of {chunk_hours} hours each")

        # Max workers limited by memory (each worker can use 3-5GB RAM)
        # With 64GB RAM, safe limit is ~8-12 workers
        cpu_based_workers = max(1, multiprocessing.cpu_count() - 4)
        max_workers = min(cpu_based_workers, warmup_config.get('max_workers', 8))
        print(f"Using {max_workers} worker processes (CPU-based: {cpu_based_workers}, memory-limited to {warmup_config.get('max_workers', 8)}).")

        total_metrics = 0
        total_stats = {
            'metrics_count': 0,
            'l1_count': 0,
            'l2_count': 0,
            'raw_logs': 0,
            'l1_sample_sum': 0,
            'l2_user_sample_sum': 0,
            'l2_route_sample_sum': 0
        }
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk[0], chunk[1], config) for chunk in chunks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, dict):
                        for key in total_stats:
                            total_stats[key] += result.get(key, 0)
                    else:
                        # Fallback for error cases returning 0
                        pass
                except Exception as exc:
                    print(f"Worker failed: {exc}")

        duration = time.time() - start_time
        print(f"\nWarmup Complete!")
        print(f"Total Unique Logs Processed: {total_stats['raw_logs']}")
        print(f"Total Metrics Generated: {total_stats['metrics_count']} (L1: {total_stats['l1_count']}, L2: {total_stats['l2_count']})")
        print(f"Total Sample Counts (Validation):")
        print(f"  - L1 (Entity): {total_stats['l1_sample_sum']}")
        print(f"  - L2 (User):   {total_stats['l2_user_sample_sum']}")
        print(f"  - L2 (Route):  {total_stats['l2_route_sample_sum']}")
        print(f"Total Time: {duration:.2f} seconds")
        print(f"Metrics per second: {total_stats['metrics_count'] / duration if duration > 0 else 0:.2f}")

    except Exception as e:
        print(f"Main execution failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
