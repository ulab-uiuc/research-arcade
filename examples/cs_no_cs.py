import psycopg2
import pandas as pd

class DatabaseAnalyzer:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            port="5433",
            dbname="postgres",
            user="cl195"
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        
        # 定义已知的分类列表
        self.known_categories = ["cs.AI", "cs.LG", "cs.CV", "cs.RO", "cs.CR", 
                                "cs.DB", "cs.DC", "cs.PF", "cs.MA", "cs.OS"]
        
        # 列名
        self.column_names = ["paper", "section", "paragraph", "figure", "table", "author"]
    
    def get_non_cs_papers(self):
        """获取没有任何cs.XX分类的论文ID列表"""
        try:
            query = """
            SELECT DISTINCT p.arxiv_id 
            FROM papers p
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc.paper_arxiv_id 
                FROM paper_category pc
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%'
            );
            """
            print(f"Debug - Getting papers with no CS categories...")
            self.cur.execute(query)
            non_cs_papers = [row[0] for row in self.cur.fetchall()]
            
            print(f"Found {len(non_cs_papers)} papers with no CS categories")
            if len(non_cs_papers) <= 10:
                print(f"Sample paper IDs: {non_cs_papers}")
            else:
                print(f"Sample paper IDs: {non_cs_papers[:10]}... (showing first 10)")
            
            return non_cs_papers
        except Exception as e:
            print(f"Error getting non-CS papers: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_non_cs_categories(self):
        """获取这些论文实际拥有的非cs分类"""
        try:
            query = """
            SELECT DISTINCT c.name, COUNT(DISTINCT p.arxiv_id) as paper_count
            FROM papers p
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc2.paper_arxiv_id 
                FROM paper_category pc2
                JOIN categories c2 ON pc2.category_id = c2.id
                WHERE c2.name LIKE 'cs.%'
            )
            AND c.name NOT LIKE 'cs.%'
            GROUP BY c.name
            ORDER BY paper_count DESC;
            """
            print(f"Debug - Getting non-CS categories for these papers...")
            self.cur.execute(query)
            results = self.cur.fetchall()
            
            categories_info = [(row[0], row[1]) for row in results]
            print(f"Non-CS categories found: {categories_info}")
            
            return categories_info
        except Exception as e:
            print(f"Error getting non-CS categories: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def execute_queries_for_non_cs(self):
        """执行查询没有任何cs.XX分类的论文的统计"""
        results = []
        
        # 首先获取没有cs分类的论文列表
        non_cs_papers = self.get_non_cs_papers()
        
        if not non_cs_papers:
            print("No papers without CS categories found, returning zeros.")
            row = ["Non-CS"] + [0] * len(self.column_names)
            results.append(row)
            return results
        
        print(f"Analyzing {len(non_cs_papers)} papers without CS categories...")
        
        # 获取这些论文的非cs分类信息
        self.get_non_cs_categories()
        
        # 定义查询语句 - 查询没有任何cs分类的论文
        statements = [
            """
            SELECT COUNT(DISTINCT p.arxiv_id) 
            FROM papers p
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc.paper_arxiv_id 
                FROM paper_category pc
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%'
            );
            """,
            """
            SELECT COUNT(*) 
            FROM sections s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc.paper_arxiv_id 
                FROM paper_category pc
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%'
            );
            """,
            """
            SELECT COUNT(*) 
            FROM paragraphs s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc.paper_arxiv_id 
                FROM paper_category pc
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%'
            );
            """,
            """
            SELECT COUNT(*) 
            FROM figures s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc.paper_arxiv_id 
                FROM paper_category pc
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%'
            );
            """,
            """
            SELECT COUNT(*) 
            FROM tables s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc.paper_arxiv_id 
                FROM paper_category pc
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%'
            );
            """,
            """
            SELECT COUNT(DISTINCT a.semantic_scholar_id) 
            FROM authors a
            JOIN paper_authors pa ON a.semantic_scholar_id = pa.author_id
            JOIN papers p ON pa.paper_arxiv_id = p.arxiv_id
            WHERE p.arxiv_id NOT IN (
                SELECT DISTINCT pc.paper_arxiv_id 
                FROM paper_category pc
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%'
            );
            """
        ]
        
        row = ["Non-CS"]  # 分类名设为 Non-CS
        
        for i, statement in enumerate(statements):
            try:
                print(f"Debug - Executing query {i+1} for non-CS papers...")
                
                self.cur.execute(statement)
                result = self.cur.fetchone()
                
                if result is None:
                    print(f"Warning: No result returned for {self.column_names[i]}")
                    count = 0
                else:
                    count = result[0] if result[0] is not None else 0
                
                row.append(count)
                print(f"✓ Non-CS - {self.column_names[i]}: {count}")
                
            except Exception as e:
                print(f"✗ Error for Non-CS - {self.column_names[i]}: {e}")
                import traceback
                traceback.print_exc()
                row.append(0)
        
        results.append(row)
        return results
    
    def execute_queries_by_category(self):
        """执行原有的按分类查询（保持兼容性）"""
        results = []
        
        statements = [
            """
            SELECT COUNT(*) FROM papers p
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE c.name = %s;
            """,
            """
            SELECT COUNT(*) FROM sections s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE c.name = %s;
            """,
            """
            SELECT COUNT(*) FROM paragraphs s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE c.name = %s;
            """,
            """
            SELECT COUNT(*) FROM figures s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE c.name = %s;
            """,
            """
            SELECT COUNT(*) FROM tables s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE c.name = %s;
            """,
            """
            SELECT COUNT(DISTINCT a.semantic_scholar_id) FROM authors a
            JOIN paper_authors pa ON a.semantic_scholar_id = pa.author_id
            JOIN papers p ON pa.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE c.name = %s;
            """
        ]
        
        for category in self.known_categories:
            row = [category]
            
            for i, statement in enumerate(statements):
                try:
                    self.cur.execute(statement, (category,))
                    count = self.cur.fetchone()[0]
                    row.append(count)
                    print(f"✓ {category} - {self.column_names[i]}: {count}")
                except Exception as e:
                    print(f"✗ Error for {category}: {e}")
                    row.append(0)
            
            results.append(row)
        
        return results
    
    def execute_all_queries(self, include_known_categories=True, include_non_cs=True):
        """执行所有查询"""
        all_results = []
        
        if include_known_categories:
            print("Querying known CS categories...")
            known_results = self.execute_queries_by_category()
            all_results.extend(known_results)
        
        if include_non_cs:
            print("\nQuerying papers with no CS categories...")
            non_cs_results = self.execute_queries_for_non_cs()
            all_results.extend(non_cs_results)
        
        return all_results
    
    def test_database_connection(self):
        """测试数据库连接和基础查询"""
        try:
            print("Testing database connection...")
            
            # 测试1: 检查总论文数
            self.cur.execute("SELECT COUNT(*) FROM papers;")
            total_papers = self.cur.fetchone()[0]
            print(f"Total papers in database: {total_papers}")
            
            # 测试2: 检查有cs分类的论文数
            self.cur.execute("""
                SELECT COUNT(DISTINCT p.arxiv_id) 
                FROM papers p
                JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
                JOIN categories c ON pc.category_id = c.id
                WHERE c.name LIKE 'cs.%';
            """)
            cs_papers = self.cur.fetchone()[0]
            print(f"Papers with CS categories: {cs_papers}")
            
            # 测试3: 检查没有cs分类的论文数
            self.cur.execute("""
                SELECT COUNT(DISTINCT p.arxiv_id) 
                FROM papers p
                WHERE p.arxiv_id NOT IN (
                    SELECT DISTINCT pc.paper_arxiv_id 
                    FROM paper_category pc
                    JOIN categories c ON pc.category_id = c.id
                    WHERE c.name LIKE 'cs.%'
                );
            """)
            non_cs_papers = self.cur.fetchone()[0]
            print(f"Papers without any CS categories: {non_cs_papers}")
            
            # 测试4: 验证总数
            print(f"Verification: {cs_papers} + {non_cs_papers} = {cs_papers + non_cs_papers} (Total: {total_papers})")
            
            # 测试5: 查看这些非cs论文的主要分类
            self.cur.execute("""
                SELECT c.name, COUNT(DISTINCT p.arxiv_id) as paper_count
                FROM papers p
                JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
                JOIN categories c ON pc.category_id = c.id
                WHERE p.arxiv_id NOT IN (
                    SELECT DISTINCT pc2.paper_arxiv_id 
                    FROM paper_category pc2
                    JOIN categories c2 ON pc2.category_id = c2.id
                    WHERE c2.name LIKE 'cs.%'
                )
                GROUP BY c.name
                ORDER BY paper_count DESC
                LIMIT 10;
            """)
            top_non_cs_categories = self.cur.fetchall()
            print(f"Top non-CS categories: {top_non_cs_categories}")
            
            return True
            
        except Exception as e:
            print(f"Database test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_dataframe(self, results):
        """创建pandas DataFrame"""
        columns = ["category"] + self.column_names
        df = pd.DataFrame(results, columns=columns)
        return df
    
    def print_table(self, df):
        """打印格式化的表格"""
        print("\n" + "="*80)
        print("Database Analysis Results (Including Non-CS Papers)")
        print("="*80)
        
        header = f"{'Category':<12}"
        for col in self.column_names:
            header += f"{col.capitalize():<12}"
        print(header)
        print("-" * 80)
        
        for _, row in df.iterrows():
            line = f"{row['category']:<12}"
            for col in self.column_names:
                line += f"{row[col]:<12}"
            print(line)
    
    def save_to_csv(self, df, filename="./csv/database_analysis_with_non_cs.csv"):
        """保存到CSV文件"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n✓ Results saved to {filename}")
    
    def run_analysis(self, mode="all"):
        """
        运行分析
        mode: "known" - 只查询已知分类
              "non_cs" - 只查询没有cs分类的论文
              "all" - 查询所有分类（默认）
        """
        try:
            print(f"Starting database analysis (mode: {mode})...")
            
            # 根据模式执行不同的查询
            if mode == "known":
                results = self.execute_all_queries(include_known_categories=True, include_non_cs=False)
            elif mode == "non_cs":
                results = self.execute_all_queries(include_known_categories=False, include_non_cs=True)
            else:  # mode == "all"
                results = self.execute_all_queries(include_known_categories=True, include_non_cs=True)
            
            df = self.create_dataframe(results)
            self.print_table(df)
            self.save_to_csv(df)
            
            print(f"\n📊 Summary:")
            print(f"Total categories analyzed: {len(df)}")
            print(f"Total papers across all categories: {df['paper'].sum():,}")
            print(f"Total authors across all categories: {df['author'].sum():,}")
            
            return df
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.close_connection()
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("✓ Database connection closed")

# 使用示例
if __name__ == "__main__":
    analyzer = DatabaseAnalyzer()
    
    # 首先测试数据库连接
    if analyzer.test_database_connection():
        print("\n" + "="*50)
        
        # 选项1：只查询没有cs分类的论文
        df = analyzer.run_analysis(mode="non_cs")
        
        # 选项2：查询所有分类（已知的cs分类 + 没有cs分类的论文）
        # df = analyzer.run_analysis(mode="all")
        
        # 选项3：只查询已知cs分类
        # df = analyzer.run_analysis(mode="known")
    else:
        print("Database connection test failed. Please check your database setup.")