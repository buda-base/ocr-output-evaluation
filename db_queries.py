"""
Database queries for fetching volume information
"""
import psycopg2
from typing import List, Dict, Tuple
from config import PGSQL_URL


def get_volumes_for_job(job_name: str) -> List[Dict[str, str]]:
    """
    Fetch all completed volumes for a given job (google_books or google_vision)
    
    Returns list of dicts with keys: w_id, i_id, i_version, volume_id
    """
    query = """
    SELECT 
        v.bdrc_w_id as w_id,
        v.bdrc_i_id as i_id,
        ENCODE(te.s3_etag, 'hex') as etag_hex,
        SUBSTRING(ENCODE(te.s3_etag, 'hex'), 1, 6) as i_version,
        v.id as volume_id
    FROM task_executions te
    JOIN volumes v ON te.volume_id = v.id
    JOIN jobs j ON te.job_id = j.id
    WHERE j.name = %s
      AND te.status = 'done'
    ORDER BY v.id
    """
    
    conn = psycopg2.connect(PGSQL_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (job_name,))
            rows = cur.fetchall()
            
            volumes = []
            for row in rows:
                volumes.append({
                    'w_id': row[0],
                    'i_id': row[1],
                    'etag_hex': row[2],
                    'i_version': row[3],
                    'volume_id': row[4]
                })
            
            return volumes
    finally:
        conn.close()


def get_all_volumes() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Fetch all completed volumes for Google Books, Google Vision, and OCRv1
    
    Returns: (gb_volumes, gv_volumes, ocrv1_volumes)
    """
    gb_volumes = get_volumes_for_job('google_books')
    gv_volumes = get_volumes_for_job('google_vision')
    ocrv1_volumes = get_volumes_for_job('ocrv1-ws-ldv1')
    
    return gb_volumes, gv_volumes, ocrv1_volumes


if __name__ == '__main__':
    # Test the database connection
    print("Testing database connection...")
    gb_vols, gv_vols, ocrv1_vols = get_all_volumes()
    print(f"Found {len(gb_vols)} Google Books volumes")
    print(f"Found {len(gv_vols)} Google Vision volumes")
    print(f"Found {len(ocrv1_vols)} OCRv1-WS-LDv1 volumes")
    
    if gb_vols:
        print(f"\nSample Google Books volume: {gb_vols[0]}")
    if gv_vols:
        print(f"Sample Google Vision volume: {gv_vols[0]}")
    if ocrv1_vols:
        print(f"Sample OCRv1 volume: {ocrv1_vols[0]}")
