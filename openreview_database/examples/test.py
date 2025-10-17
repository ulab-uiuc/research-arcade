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
        cur.execute("ALTER TABLE paragraphs RENAME TO openreview_paragraphs;")
        print("✓ 表已重命名: paragraphs → openreview_paragraphs")
        
        # 重命名相关对象（如果存在）
        try:
            cur.execute("ALTER INDEX IF EXISTS paragraphs_pkey RENAME TO openreview_paragraphs_pkey;")
            cur.execute("ALTER SEQUENCE IF EXISTS paragraphs_id_seq RENAME TO openreview_paragraphs_id_seq;")
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