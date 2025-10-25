import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


async def start_background_worker(service, vector_engine, config: Dict[str, Any], interval: int = 30):
    """Start the background worker as an asyncio Task.

    The worker will:
    - Process `prototype_recompute_queue`: recompute prototypes for queued categories.
    - Process `vector_deletion_queue`: when deletions exist, rebuild FAISS index and register it.

    Blocking operations (model embedding / index build) are offloaded to a thread via asyncio.to_thread.
    """
    stop_event = asyncio.Event()

    async def _loop():
        logger.info("Prototype worker started (interval=%s seconds)", interval)
        while not stop_event.is_set():
            try:
                # Process prototype recompute queue
                try:
                    with service.db_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT category_id FROM prototype_recompute_queue ORDER BY last_updated")
                            rows = cur.fetchall()

                    category_ids = [r[0] for r in rows] if rows else []

                    for cat_id in category_ids:
                        try:
                            logger.info(f"Recomputing prototype for category {cat_id}")
                            # Offload heavy computation to thread
                            await asyncio.to_thread(service.prototype_manager.create_prototype_for_category, cat_id)
                            # Remove from queue on success
                            with service.db_manager.get_connection() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("DELETE FROM prototype_recompute_queue WHERE category_id = %s", (cat_id,))
                                    conn.commit()
                            logger.info(f"Prototype recompute completed for category {cat_id}")
                        except Exception as e:
                            logger.exception(f"Failed to recompute prototype for category {cat_id}: {e}")
                except Exception:
                    logger.exception("Error reading prototype_recompute_queue")

                # Process vector deletion queue: trigger a FAISS rebuild if there are deletions
                try:
                    with service.db_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT image_id FROM vector_deletion_queue ORDER BY queued_at")
                            del_rows = cur.fetchall()

                    to_delete_ids = [r[0] for r in del_rows] if del_rows else []

                    if to_delete_ids:
                        logger.info(f"Found {len(to_delete_ids)} vector deletions queued — rebuilding FAISS index")

                        # Build new FAISS index from current images
                        images = service.image_manager.get_all_images()
                        images_data = []
                        for img in images:
                            img_data = img['image_data']
                            if isinstance(img_data, memoryview):
                                img_data = img_data.tobytes()
                            images_data.append((img['image_id'], img_data))

                        index_path = config.get('faiss_index_path', 'data/faiss_index.bin')
                        index_type = config.get('faiss_index_type', 'flatl2')

                        # Ensure directory exists
                        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

                        # Offload index build to thread (it's CPU/GPU heavy)
                        try:
                            faiss_index = await asyncio.to_thread(vector_engine.build_index, images_data, index_type, index_path)

                            # Register index in DB and update vector index_ids
                            index_id = service.faiss_index_manager.register_index(index_type, str(index_path))
                            service.faiss_index_manager.update_vector_index_ids(index_id)

                            # Clear deletion queue
                            with service.db_manager.get_connection() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("DELETE FROM vector_deletion_queue")
                                    conn.commit()

                            logger.info(f"FAISS index rebuilt and registered (index_id={index_id})")
                        except Exception as e:
                            logger.exception(f"Failed to rebuild FAISS index: {e}")
                except Exception:
                    logger.exception("Error reading vector_deletion_queue")

            except Exception:
                logger.exception("Unexpected error in prototype worker loop")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

        logger.info("Prototype worker stopped")

    task = asyncio.create_task(_loop())

    # Return a stop function that can be awaited to stop the loop
    async def stop():
        stop_event.set()
        await task

    return stop
