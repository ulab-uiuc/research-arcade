import psycopg2

def rename_table():
    conn = psycopg2.connect(
        host="localhost",
        dbname="iclr_openreview_database",
        user="jingjunx",
        password="",
        port="5432"
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    try:
        # 重命名表
        cur.execute("ALTER TABLE papers_reviews RENAME TO openreview_papers_reviews;")
        print("✓ 表已重命名: papers_reviews → openreview_papers_reviews")
        
        # 重命名相关对象（如果存在）
        try:
            cur.execute("ALTER INDEX IF EXISTS papers_reviews_pkey RENAME TO openreview_papers_reviews_pkey;")
            cur.execute("ALTER SEQUENCE IF EXISTS papers_reviews_id_seq RENAME TO openreview_papers_reviews_id_seq;")
            print("✓ 索引和序列也已重命名")
        except:
            pass
            
    except Exception as e:
        print(f"✗ 错误: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    rename_table()