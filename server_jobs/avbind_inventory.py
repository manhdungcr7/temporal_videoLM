"""Read available caches and cross-domain assets; no inference, no labels."""
ids=(1454,2396,1727,1647,2345,2348,2141,2139,2327,1603,2097,1532,1580,2040,1666,2124)
daily=Path('/kaggle/working/daily_omni/videos')
RESULT=dict(server=LAB_SERVER,physical_cache_ids=[i for i in ids if (OUT/f'cq_physical_memory_{i}.json').exists()],
            daily_videos=[str(p) for p in daily.rglob('*') if p.suffix.lower() in ('.mp4','.mkv','.webm','.mov')][:5],
            output_files=[dict(name=str(p.relative_to(OUT)),size=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and ('avbind' in p.name or p.suffix=='.pt')][:100],
            source_metadata=globals().get('CQ_INFERENCE_METADATA'))
print('AVBIND_INVENTORY',json.dumps(RESULT),flush=True)
