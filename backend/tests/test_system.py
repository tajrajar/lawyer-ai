import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    print('Test 1: Imports...')
    try:
        from data.pdf_engine import extract_text_auto
        from data.indexing.dedup import chunk_text
        from core.citation_guard import guard
        from core.router import call_llm
        print('  OK')
        return True
    except ImportError as e:
        print(f'  FAIL: {e}')
        return False

def test_guard():
    print('Test 2: Citation guard...')
    from core.citation_guard import is_valid_ref
    assert is_valid_ref('PPC 420')
    assert not is_valid_ref('PPC 9999')
    assert is_valid_ref('Article 199')
    assert is_valid_ref('Section 489-F')
    assert is_valid_ref('Family Courts Act 1964')
    print('  OK')
    return True

def test_vectorstore():
    print('Test 3: Vector store...')
    files = ['data/vector_store/law.faiss', 'data/vector_store/chunks.pkl']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f'  WARN: Run law_indexer.py first')
        return False
    print('  OK')
    return True

if __name__ == '__main__':
    print('=' * 40)
    r = [test_imports(), test_guard(), test_vectorstore()]
    print(f'{sum(r)}/{len(r)} tests passed')
