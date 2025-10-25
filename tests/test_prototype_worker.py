import asyncio
import pytest
import sys
from pathlib import Path

# Ensure project root is on sys.path so `backend` package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.prototype_worker import start_background_worker


class FakeDBManager:
    def __init__(self, prototype_queue=None, deletion_queue=None):
        self.prototype_queue = list(prototype_queue or [])
        self.deletion_queue = list(deletion_queue or [])

    class _Conn:
        def __init__(self, outer):
            self.outer = outer

        def cursor(self):
            outer = self.outer

            class Cur:
                def __init__(self, outer):
                    self.outer = outer
                    self.last_query = None
                    self.last_params = None

                def execute(self, query, params=None):
                    # store query for fetchall/delete handling
                    self.last_query = query.lower()
                    self.last_params = params

                    # Handle deletes
                    if query.strip().lower().startswith('delete from prototype_recompute_queue') and params:
                        cid = params[0]
                        if cid in self.outer.prototype_queue:
                            self.outer.prototype_queue.remove(cid)
                    if query.strip().lower().startswith('delete from vector_deletion_queue'):
                        # clear entire deletion queue
                        self.outer.deletion_queue.clear()

                def fetchall(self):
                    if 'select category_id from prototype_recompute_queue' in (self.last_query or ''):
                        return [(cid,) for cid in list(self.outer.prototype_queue)]
                    if 'select image_id from vector_deletion_queue' in (self.last_query or ''):
                        return [(iid,) for iid in list(self.outer.deletion_queue)]
                    return []

                def close(self):
                    return

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Cur(outer)

        def commit(self):
            return

    def get_connection(self):
        # Provide a context manager compatible with worker expectations
        outer = self

        class Ctx:
            def __enter__(self_inner):
                return FakeDBManager._Conn(outer)

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return Ctx()


class FakePrototypeManager:
    def __init__(self):
        self.recomputed = []

    def create_prototype_for_category(self, category_id):
        # simulate work
        self.recomputed.append(category_id)
        return [0.0]  # dummy vector


class FakeFaissManager:
    def __init__(self):
        self.registered = []

    def register_index(self, index_type, index_filepath):
        idx = len(self.registered) + 1
        self.registered.append((idx, index_type, index_filepath))
        return idx

    def update_vector_index_ids(self, index_id):
        self.updated_index_id = index_id


class FakeImageManager:
    def __init__(self, images):
        self._images = images

    def get_all_images(self):
        return self._images


class FakeService:
    def __init__(self, prototype_q=None, deletion_q=None, images=None):
        self.db_manager = FakeDBManager(prototype_q, deletion_q)
        self.prototype_manager = FakePrototypeManager()
        self.faiss_index_manager = FakeFaissManager()
        self.image_manager = FakeImageManager(images or [])


class FakeVectorEngine:
    def __init__(self):
        self.built = False

    def build_index(self, images_data, index_type, save_path):
        # pretend to build index
        self.built = True

        class Index:
            def __init__(self, ntotal):
                self.ntotal = ntotal

        return Index(len(images_data))


def test_prototype_recompute_queue_processing():
    async def runner():
        # seed prototype queue with category 1
        service = FakeService(prototype_q=[1], images=[{'image_id': 1, 'image_data': b'img'}])
        vector_engine = FakeVectorEngine()
        config = {'faiss_index_path': 'data/faiss_test.bin', 'faiss_index_type': 'flatl2'}

        # start worker with short interval
        stop = await start_background_worker(service, vector_engine, config, interval=0.5)

        # allow some time for the worker to process
        await asyncio.sleep(1.2)

        # stop worker
        await stop()

        # prototype should have been recomputed and removed from queue
        assert 1 in service.prototype_manager.recomputed
        assert service.db_manager.prototype_queue == []

    asyncio.run(runner())


def test_vector_deletion_queue_triggers_faiss_rebuild():
    async def runner():
        # seed deletion queue and images
        service = FakeService(prototype_q=[], deletion_q=[10], images=[{'image_id': 2, 'image_data': b'img2'}, {'image_id': 3, 'image_data': b'img3'}])
        vector_engine = FakeVectorEngine()
        config = {'faiss_index_path': 'data/faiss_test.bin', 'faiss_index_type': 'flatl2'}

        stop = await start_background_worker(service, vector_engine, config, interval=0.5)

        await asyncio.sleep(1.5)

        await stop()

        # vector engine should have built index and faiss manager should have registered it
        assert vector_engine.built is True
        assert len(service.faiss_index_manager.registered) >= 1
        assert service.db_manager.deletion_queue == []

    asyncio.run(runner())
