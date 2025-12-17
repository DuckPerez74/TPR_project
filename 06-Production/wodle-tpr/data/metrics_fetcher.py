"""
Metrics Fetcher Module

This module provides functions to fetch L1 and L2 metrics from OpenSearch
for training purposes. Includes both entity-specific and bulk fetch operations.
"""

import sys
import numpy as np
from constants import L1_FEATURE_ORDER, L2_USER_FEATURES, L2_ROUTE_FEATURES


def get_unique_entities(client, index_name, start_date=None, end_date=None):
    """
    Get list of unique entity IDs from the metrics index.

    Args:
        client: OpenSearch client
        index_name: Index pattern (e.g., 'metrics-tpr*')
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        List of entity IDs
    """
    print(f"  Getting unique entities from {index_name}...")

    query = {
        "size": 0,
        "aggs": {
            "unique_entities": {
                "terms": {
                    "field": "entity_id.keyword",
                    "size": 10000
                }
            }
        }
    }

    if start_date and end_date:
        query["query"] = {
            "range": {
                "@timestamp": {
                    "gte": start_date.isoformat(),
                    "lt": end_date.isoformat()
                }
            }
        }

    try:
        response = client.search(index=index_name, body=query, ignore_unavailable=True)
        buckets = response.get('aggregations', {}).get('unique_entities', {}).get('buckets', [])
        entities = [bucket['key'] for bucket in buckets]
        print(f"    Found {len(entities)} unique entities")
        return entities
    except Exception as e:
        print(f"    ERROR: Failed to get unique entities: {str(e)}")
        return []


def fetch_entity_l1_metrics_all_windows(client, index_name, entity_id, start_date, end_date, windows_list):
    """
    Fetch L1 metrics for a specific entity across MULTIPLE observation windows.
    More efficient than fetching each window separately.

    Args:
        client: OpenSearch client
        index_name: Index pattern
        entity_id: Entity ID to fetch
        start_date: Start datetime
        end_date: End datetime
        windows_list: List of window sizes in minutes (e.g., [60, 30, 10])

    Returns:
        Dict mapping window_minutes to numpy array of samples
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"entity_id.keyword": entity_id}},
                    {"term": {"layer.keyword": "L1"}},
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_date.isoformat(),
                                "lt": end_date.isoformat()
                            }
                        }
                    }
                ]
            }
        },
        "size": 10000,
        "sort": [{"@timestamp": "asc"}]
    }

    try:
        all_docs = []
        response = client.search(index=index_name, body=query, scroll='2m', ignore_unavailable=True)
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        all_docs.extend(hits)

        # Fetch ALL documents without limit to ensure complete data collection
        while hits:
            response = client.scroll(scroll_id=scroll_id, scroll='2m')
            hits = response['hits']['hits']
            all_docs.extend(hits)
            scroll_id = response.get('_scroll_id')

        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except:
                pass

        window_metrics = {w: [] for w in windows_list}

        for doc in all_docs:
            source = doc['_source']
            window = source.get('observation_window')
            if window in windows_list:
                metrics = source.get('metrics', {})
                sample = [metrics.get(feat, 0) for feat in L1_FEATURE_ORDER]
                window_metrics[window].append(sample)

        for window in window_metrics:
            window_metrics[window] = np.array(window_metrics[window]) if window_metrics[window] else np.array([])

        return window_metrics

    except Exception as e:
        print(f"    ERROR: Failed to fetch L1 for entity {entity_id}: {str(e)}")
        return {w: np.array([]) for w in windows_list}


def fetch_entity_l1_metrics(client, index_name, entity_id, start_date, end_date, window_minutes=None):
    """
    Fetch L1 metrics for a specific entity.

    Args:
        client: OpenSearch client
        index_name: Index pattern
        entity_id: Entity ID to fetch
        start_date: Start datetime
        end_date: End datetime
        window_minutes: Optional specific window (if None, fetches all windows)

    Returns:
        List of metric samples (numpy arrays)
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"entity_id.keyword": entity_id}},
                    {"term": {"layer.keyword": "L1"}},
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_date.isoformat(),
                                "lt": end_date.isoformat()
                            }
                        }
                    }
                ]
            }
        },
        "size": 10000,
        "sort": [{"@timestamp": "asc"}]
    }

    if window_minutes is not None:
        query["query"]["bool"]["must"].append({"term": {"observation_window": window_minutes}})

    try:
        all_docs = []
        response = client.search(index=index_name, body=query, scroll='2m', ignore_unavailable=True)
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        all_docs.extend(hits)

        # Fetch ALL documents without limit to ensure complete data collection
        while hits:
            response = client.scroll(scroll_id=scroll_id, scroll='2m')
            hits = response['hits']['hits']
            all_docs.extend(hits)
            scroll_id = response.get('_scroll_id')

        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except:
                pass

        samples = []
        for doc in all_docs:
            metrics = doc['_source'].get('metrics', {})
            sample = [metrics.get(feat, 0) for feat in L1_FEATURE_ORDER]
            samples.append(sample)

        return np.array(samples) if samples else np.array([])

    except Exception as e:
        print(f"    ERROR: Failed to fetch L1 for entity {entity_id}: {str(e)}")
        return np.array([])


def fetch_entity_l2_metrics_all(client, index_name, entity_id, start_date, end_date, dimensions_list, windows_list):
    """
    Fetch L2 metrics for a specific entity across ALL dimensions and windows.
    More efficient than fetching each combination separately.

    Args:
        client: OpenSearch client
        index_name: Index pattern
        entity_id: Entity ID
        start_date: Start datetime
        end_date: End datetime
        dimensions_list: List of dimensions (e.g., ['user', 'route'])
        windows_list: List of window sizes in minutes (e.g., [60, 30, 10])

    Returns:
        Dict with structure: {dimension: {window: {dimension_value: samples_array}}}
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"entity_id.keyword": entity_id}},
                    {"term": {"layer.keyword": "L2"}},
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_date.isoformat(),
                                "lt": end_date.isoformat()
                            }
                        }
                    }
                ]
            }
        },
        "size": 10000,
        "sort": [{"@timestamp": "asc"}]
    }

    try:
        all_docs = []
        response = client.search(index=index_name, body=query, scroll='2m', ignore_unavailable=True)
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        all_docs.extend(hits)

        # Fetch ALL documents without limit to ensure complete data collection
        while hits:
            response = client.scroll(scroll_id=scroll_id, scroll='2m')
            hits = response['hits']['hits']
            all_docs.extend(hits)
            scroll_id = response.get('_scroll_id')

        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except:
                pass

        result = {}
        for dim in dimensions_list:
            result[dim] = {}
            for window in windows_list:
                result[dim][window] = {}

        for doc in all_docs:
            source = doc['_source']
            dimension = source.get('dimension')
            window = source.get('observation_window')
            dim_value = source.get('dimension_value')

            if dimension in dimensions_list and window in windows_list and dim_value:
                feature_order = L2_USER_FEATURES if dimension == 'user' else L2_ROUTE_FEATURES
                metrics = source.get('metrics', {})

                sample = [metrics.get(feat, 0) for feat in feature_order]

                if dim_value not in result[dimension][window]:
                    result[dimension][window][dim_value] = []
                result[dimension][window][dim_value].append(sample)

        for dim in result:
            for window in result[dim]:
                for dim_value in result[dim][window]:
                    result[dim][window][dim_value] = np.array(result[dim][window][dim_value])

        return result

    except Exception as e:
        print(f"    ERROR: Failed to fetch L2 for entity {entity_id}: {str(e)}")
        result = {}
        for dim in dimensions_list:
            result[dim] = {}
            for window in windows_list:
                result[dim][window] = {}
        return result


def fetch_entity_l2_metrics(client, index_name, entity_id, start_date, end_date, dimension, window_minutes=None):
    """
    Fetch L2 metrics for a specific entity and dimension.

    Args:
        client: OpenSearch client
        index_name: Index pattern
        entity_id: Entity ID
        start_date: Start datetime
        end_date: End datetime
        dimension: 'user' or 'route'
        window_minutes: Optional specific window

    Returns:
        Dict mapping dimension_value to list of samples
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"entity_id.keyword": entity_id}},
                    {"term": {"layer.keyword": "L2"}},
                    {"term": {"dimension.keyword": dimension}},
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_date.isoformat(),
                                "lt": end_date.isoformat()
                            }
                        }
                    }
                ]
            }
        },
        "size": 10000,
        "sort": [{"@timestamp": "asc"}]
    }

    if window_minutes is not None:
        query["query"]["bool"]["must"].append({"term": {"observation_window": window_minutes}})

    try:
        all_docs = []
        response = client.search(index=index_name, body=query, scroll='2m', ignore_unavailable=True)
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        all_docs.extend(hits)

        # Fetch ALL documents without limit to ensure complete data collection
        while hits:
            response = client.scroll(scroll_id=scroll_id, scroll='2m')
            hits = response['hits']['hits']
            all_docs.extend(hits)
            scroll_id = response.get('_scroll_id')

        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except:
                pass

        feature_order = L2_USER_FEATURES if dimension == 'user' else L2_ROUTE_FEATURES
        dimension_metrics = {}

        for doc in all_docs:
            source = doc['_source']
            dim_value = source.get('dimension_value', 'unknown')
            metrics = source.get('metrics', {})

            sample = [metrics.get(feat, 0) for feat in feature_order]

            if dim_value not in dimension_metrics:
                dimension_metrics[dim_value] = []
            dimension_metrics[dim_value].append(sample)

        for dim_value in dimension_metrics:
            dimension_metrics[dim_value] = np.array(dimension_metrics[dim_value])

        return dimension_metrics

    except Exception as e:
        print(f"    ERROR: Failed to fetch L2 {dimension} for entity {entity_id}: {str(e)}")
        return {}


def fetch_l1_metrics_from_opensearch(client, index_name, window_minutes=60, min_samples=100, max_docs=1000000):
    """
    Fetch L1 metrics for a specific observation window (bulk fetch).

    Args:
        client: OpenSearch client
        index_name: Index pattern
        window_minutes: Observation window size in minutes
        min_samples: Minimum samples required per entity
        max_docs: Maximum documents to fetch

    Returns:
        Dict mapping entity_id to list of samples
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"layer.keyword": "L1"}},
                    {"term": {"observation_window": window_minutes}}
                ]
            }
        },
        "size": 10000
    }

    print(f"  Fetching L1 metrics (window={window_minutes}min)...")
    print(f"    Index pattern: {index_name}")

    try:
        count_response = client.count(index=index_name, ignore_unavailable=True)
        total_docs = count_response.get('count', 0)
        print(f"    Total documents in index: {total_docs:,}")
    except Exception as e:
        print(f"    WARNING: Could not count documents: {str(e)}")

    try:
        l1_count_query = {"query": {"term": {"layer.keyword": "L1"}}}
        l1_count = client.count(index=index_name, body=l1_count_query, ignore_unavailable=True).get('count', 0)
        print(f"    L1 documents in index: {l1_count:,}")
    except Exception as e:
        print(f"    WARNING: Could not count L1 documents: {str(e)}")

    all_docs = []
    scroll_id = None

    try:
        response = client.search(index=index_name, body=query, scroll='2m', ignore_unavailable=True)
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']

        all_docs.extend(hits)

        while hits and len(all_docs) < max_docs:
            response = client.scroll(scroll_id=scroll_id, scroll='2m')
            hits = response['hits']['hits']
            all_docs.extend(hits)
            scroll_id = response.get('_scroll_id')

            if len(all_docs) >= max_docs:
                print(f"    WARNING: Reached max document limit ({max_docs}), stopping fetch", file=sys.stderr)
                break

    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception as e:
                print(f"    WARNING: Failed to clear scroll context: {str(e)}", file=sys.stderr)

    print(f"    Found {len(all_docs)} metric documents")

    entity_metrics = {}

    for doc in all_docs:
        source = doc['_source']
        entity_id = source.get('entity_id')
        metrics = source.get('metrics', {})

        if not entity_id:
            continue

        vector = [metrics.get(feature, 0) for feature in L1_FEATURE_ORDER]

        if entity_id not in entity_metrics:
            entity_metrics[entity_id] = []

        entity_metrics[entity_id].append(vector)

    entity_metrics_filtered = {
        eid: samples for eid, samples in entity_metrics.items()
        if len(samples) >= min_samples
    }

    print(f"    Entities with >={min_samples} samples: {len(entity_metrics_filtered)}")

    return entity_metrics_filtered


def fetch_l2_metrics_from_opensearch(client, index_name, dimension, window_minutes=60, min_samples=20, max_docs=1000000):
    """
    Fetch L2 metrics for a specific dimension (bulk fetch).

    Args:
        client: OpenSearch client
        index_name: Index pattern
        dimension: 'user' or 'route'
        window_minutes: Observation window size in minutes
        min_samples: Minimum samples required per dimension value
        max_docs: Maximum documents to fetch

    Returns:
        Dict mapping dimension_value to list of samples
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"layer.keyword": "L2"}},
                    {"term": {"dimension.keyword": dimension}},
                    {"term": {"observation_window": window_minutes}}
                ]
            }
        },
        "size": 10000
    }

    print(f"  Fetching L2 metrics (dimension={dimension}, window={window_minutes}min)...")
    print(f"    Index pattern: {index_name}")

    try:
        l2_count_query = {"query": {"bool": {"must": [{"term": {"layer.keyword": "L2"}}, {"term": {"dimension.keyword": dimension}}]}}}
        l2_count = client.count(index=index_name, body=l2_count_query, ignore_unavailable=True).get('count', 0)
        print(f"    L2 {dimension} documents in index: {l2_count:,}")
    except Exception as e:
        print(f"    WARNING: Could not count L2 documents: {str(e)}")

    all_docs = []
    scroll_id = None

    try:
        response = client.search(index=index_name, body=query, scroll='2m', ignore_unavailable=True)
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']

        all_docs.extend(hits)

        while hits and len(all_docs) < max_docs:
            response = client.scroll(scroll_id=scroll_id, scroll='2m')
            hits = response['hits']['hits']
            all_docs.extend(hits)
            scroll_id = response.get('_scroll_id')

            if len(all_docs) >= max_docs:
                print(f"    WARNING: Reached max document limit ({max_docs}), stopping fetch", file=sys.stderr)
                break

    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception as e:
                print(f"    WARNING: Failed to clear scroll context: {str(e)}", file=sys.stderr)

    print(f"    Found {len(all_docs)} metric documents")

    feature_order = L2_USER_FEATURES if dimension == 'user' else L2_ROUTE_FEATURES

    dimension_metrics = {}

    for doc in all_docs:
        source = doc['_source']
        dimension_value = source.get('dimension_value')
        metrics = source.get('metrics', {})

        if not dimension_value:
            continue

        vector = []
        for feature in feature_order:
            value = metrics.get(feature, 0)
            if isinstance(value, dict):
                continue
            vector.append(value if isinstance(value, (int, float)) else 0)

        if dimension_value not in dimension_metrics:
            dimension_metrics[dimension_value] = []

        dimension_metrics[dimension_value].append(vector)

    dimension_metrics_filtered = {
        dim_val: samples for dim_val, samples in dimension_metrics.items()
        if len(samples) >= min_samples
    }

    print(f"    {dimension.capitalize()}s with >={min_samples} samples: {len(dimension_metrics_filtered)}")

    return dimension_metrics_filtered
