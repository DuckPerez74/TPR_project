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

def process_chunk_by_entity(chunk_start, chunk_end, config):
    pid = os.getpid()
    print(f"[PID {pid}] Starting chunk: {chunk_start} to {chunk_end}")

    try:
        client = OpenSearchClient.get_instance(config)

        fetcher = LogFetcher(client, config)
        preprocessor = DataPreprocessor(config)
        l1_calc = L1MetricsCalculator(config)
        l2_calc = L2MetricsCalculator(config, client)
        storage = MetricsStorage(client, config)

        fetch_end = chunk_end

        # STEP 1: Get list of active entities (fast aggregation query)
        # NOTE: Uses chunk time range (NOT lookback) to get entities active in this chunk
        print(f"[PID {pid}] Getting active entities...")
        active_entities = fetcher.get_active_entities(chunk_start, fetch_end)

        if not active_entities:
            print(f"[PID {pid}] No active entities in chunk")
            return {'metrics_count': 0, 'l1_count': 0, 'l2_count': 0, 'raw_logs': 0, 'l1_sample_sum': 0, 'l2_user_sample_sum': 0, 'l2_route_sample_sum': 0}

        print(f"[PID {pid}] Found {len(active_entities)} active entities")

        # Statistics
        stats = {
            'metrics_count': 0,
            'l1_count': 0,
            'l2_count': 0,
            'raw_logs': 0,
            'l1_sample_sum': 0,
            'l2_user_sample_sum': 0,
            'l2_route_sample_sum': 0
        }

        metrics_buffer = []
        BULK_SIZE = 5000

        active_windows = config.get('observation_windows', {}).get('enabled', [10, 30, 60])
        l2_enabled = config.get('metrics', {}).get('layers', {}).get('L2', {}).get('enabled', False)

        # IMPORTANT: Use max window for lookback to ensure data availability
        max_window = max(active_windows)
        lookback_time = chunk_start - timedelta(minutes=max_window)

        # STEP 2: Process each entity independently
        for idx, entity_id in enumerate(active_entities):
            try:
                # Validate entity_id
                if not entity_id or entity_id in ['-', '', None]:
                    continue

                # Fetch logs for this entity only (with lookback)
                entity_df = fetcher.fetch_logs_by_entity(entity_id, lookback_time, fetch_end)

                if entity_df.empty:
                    continue

                # Prepare data
                entity_df = preprocessor.prepare(entity_df)

                if entity_df.empty:
                    continue

                # FIXED: Count only logs within chunk boundaries (not lookback)
                chunk_logs = entity_df[(entity_df['@timestamp'] >= chunk_start) &
                                       (entity_df['@timestamp'] < chunk_end)]
                stats['raw_logs'] += len(chunk_logs)
                del chunk_logs

                # Calculate metrics for all windows
                for window in active_windows:
                    freq = f"{window}min"
                    aligned_times = pd.date_range(start=chunk_start, end=chunk_end, freq=freq, inclusive='left')

                    for current_time in aligned_times:
                        current_time = current_time.to_pydatetime()

                        window_start = current_time - timedelta(minutes=window)

                        # Slice window data
                        window_df = entity_df[(entity_df['@timestamp'] >= window_start) &
                                              (entity_df['@timestamp'] < current_time)]

                        if window_df.empty:
                            del window_df
                            continue

                        # L1 Calculation
                        try:
                            l1_metrics = l1_calc.calculate(window_df, window)

                            doc = {
                                '@timestamp': current_time.isoformat(),
                                'window_start_time': window_start.isoformat(),
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

                            del l1_metrics
                        except Exception as e:
                            print(f"[PID {pid}] ERROR in L1 for entity {entity_id}, window {window}min @ {current_time}: {str(e)}")

                        # L2 Calculation
                        if l2_enabled:
                            try:
                                for dimension in l2_calc.dimensions:
                                    dim_results = l2_calc.calculate(window_df, dimension)
                                    for result in dim_results:
                                        doc = {
                                            '@timestamp': current_time.isoformat(),
                                            'window_start_time': window_start.isoformat(),
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

                                    del dim_results
                            except Exception as e:
                                print(f"[PID {pid}] ERROR in L2 for entity {entity_id}, window {window}min @ {current_time}: {str(e)}")

                        # Flush buffer if full
                        if len(metrics_buffer) >= BULK_SIZE:
                            storage.save_metrics_bulk(metrics_buffer)
                            stats['metrics_count'] += len(metrics_buffer)
                            metrics_buffer = []
                            gc.collect()

                        del window_df

                    del aligned_times

                # Free entity DataFrame and force GC
                del entity_df
                gc.collect()

                # Progress update every 50 entities
                if (idx + 1) % 50 == 0:
                    progress_pct = ((idx + 1) / len(active_entities)) * 100
                    print(f"[PID {pid}] Progress: {progress_pct:.1f}% ({idx + 1}/{len(active_entities)} entities, {stats['raw_logs']:,} logs)")

            except Exception as e:
                print(f"[PID {pid}] ERROR processing entity {entity_id}: {str(e)}")
                traceback.print_exc()
                continue

        # Final flush
        if metrics_buffer:
            storage.save_metrics_bulk(metrics_buffer)
            stats['metrics_count'] += len(metrics_buffer)

        # Cleanup
        if l2_enabled and l2_calc:
            l2_calc.cleanup()

        del metrics_buffer, l1_calc, l2_calc, storage, fetcher, preprocessor
        gc.collect()

        print(f"[PID {pid}] Finished {chunk_start}. Saved {stats['metrics_count']} metrics (L1: {stats['l1_count']}, L2: {stats['l2_count']}).")
        print(f"[PID {pid}] Processed {stats['raw_logs']:,} logs from {len(active_entities)} entities")
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

        print(f"WARMUP STARTED (OPTIMIZED BY-ENTITY): {start_date} to {end_date}")
        print("Processing entities independently with parallel workers...")

        # Split into chunks
        chunk_hours = warmup_config.get('chunk_hours', 3)
        chunks = []
        curr = start_date
        while curr < end_date:
            next_chunk = min(curr + timedelta(hours=chunk_hours), end_date)
            chunks.append((curr, next_chunk))
            curr = next_chunk

            if curr >= end_date:
                break

        print(f"Processing {len(chunks)} chunks of {chunk_hours} hours each")

        # Max workers - with by-entity processing, we can use more workers
        cpu_based_workers = max(1, multiprocessing.cpu_count() - 2)
        max_workers = min(cpu_based_workers, warmup_config.get('max_workers', 8))
        print(f"Using {max_workers} worker processes")

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
            futures = [executor.submit(process_chunk_by_entity, chunk[0], chunk[1], config) for chunk in chunks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, dict):
                        for key in total_stats:
                            total_stats[key] += result.get(key, 0)
                except Exception as exc:
                    print(f"Worker failed: {exc}")

        duration = time.time() - start_time
        print(f"\nWarmup Complete!")
        print(f"Total Unique Logs Processed: {total_stats['raw_logs']:,}")
        print(f"Total Metrics Generated: {total_stats['metrics_count']:,} (L1: {total_stats['l1_count']:,}, L2: {total_stats['l2_count']:,})")
        print(f"Total Sample Counts (Validation):")
        print(f"  - L1 (Entity): {total_stats['l1_sample_sum']:,}")
        print(f"  - L2 (User):   {total_stats['l2_user_sample_sum']:,}")
        print(f"  - L2 (Route):  {total_stats['l2_route_sample_sum']:,}")
        print(f"Total Time: {duration:.2f} seconds ({duration/3600:.2f} hours)")
        print(f"Metrics per second: {total_stats['metrics_count'] / duration if duration > 0 else 0:.2f}")

    except Exception as e:
        print(f"Main execution failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
