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
    
    def get_other_cs_categories(self):
        """获取数据库中所有其他的cs.XX分类"""
        try:
            # 使用分别的查询而不是IN子句
            query = """
            SELECT DISTINCT c.name FROM categories c
            WHERE c.name LIKE 'cs.%'
            ORDER BY c.name;
            """
            print(f"Debug - Getting all CS categories...")
            self.cur.execute(query)
            all_cs_categories = [row[0] for row in self.cur.fetchall()]
            
            # 在Python中过滤，而不是在SQL中
            other_categories = [cat for cat in all_cs_categories if cat not in self.known_categories]
            
            print(f"All CS categories: {all_cs_categories}")
            print(f"Known categories: {self.known_categories}")
            print(f"Other categories: {other_categories}")
            
            return other_categories
        except Exception as e:
            print(f"Error getting other CS categories: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def execute_queries_for_other_cs(self):
        """执行查询其他cs.XX分类的语句 - 使用OR条件而不是NOT IN"""
        results = []
        
        # 首先获取其他分类列表
        other_categories = self.get_other_cs_categories()
        
        if not other_categories:
            print("No other CS categories found, returning zeros.")
            row = ["cs.Others"] + [0] * len(self.column_names)
            results.append(row)
            return results
        
        print(f"Found {len(other_categories)} other categories: {other_categories}")
        
        # 构建OR条件字符串
        or_conditions = " OR ".join([f"c.name = %s" for _ in other_categories])
        
        # 定义查询语句 - 使用OR而不是NOT IN
        statements = [
            f"""
            SELECT COUNT(DISTINCT p.arxiv_id) FROM papers p
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE ({or_conditions});
            """,
            f"""
            SELECT COUNT(*) FROM sections s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE ({or_conditions});
            """,
            f"""
            SELECT COUNT(*) FROM paragraphs s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE ({or_conditions});
            """,
            f"""
            SELECT COUNT(*) FROM figures s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE ({or_conditions});
            """,
            f"""
            SELECT COUNT(*) FROM tables s
            JOIN papers p ON s.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE ({or_conditions});
            """,
            f"""
            SELECT COUNT(DISTINCT a.semantic_scholar_id) FROM authors a
            JOIN paper_authors pa ON a.semantic_scholar_id = pa.author_id
            JOIN papers p ON pa.paper_arxiv_id = p.arxiv_id
            JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
            JOIN categories c ON pc.category_id = c.id
            WHERE ({or_conditions});
            """
        ]
        
        row = ["cs.Others"]  # 分类名设为 cs.Others
        
        for i, statement in enumerate(statements):
            try:
                print(f"Debug - Executing query {i+1}: {statement}")
                print(f"Debug - Parameters: {other_categories}")
                print(f"Debug - Parameters length: {len(other_categories)}")
                
                self.cur.execute(statement, other_categories)
                result = self.cur.fetchone()
                
                if result is None:
                    print(f"Warning: No result returned for {self.column_names[i]}")
                    count = 0
                else:
                    count = result[0] if result[0] is not None else 0
                
                row.append(count)
                print(f"✓ cs.Others - {self.column_names[i]}: {count}")
                
            except Exception as e:
                print(f"✗ Error for cs.Others - {self.column_names[i]}: {e}")
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
    
    def execute_all_queries(self, include_known_categories=True, include_other_cs=True):
        """执行所有查询"""
        all_results = []
        
        if include_known_categories:
            print("Querying known CS categories...")
            known_results = self.execute_queries_by_category()
            all_results.extend(known_results)
        
        if include_other_cs:
            print("\nQuerying other CS categories...")
            other_results = self.execute_queries_for_other_cs()
            all_results.extend(other_results)
        
        return all_results
    
    def test_database_connection(self):
        """测试数据库连接和基础查询"""
        try:
            print("Testing database connection...")
            
            # 测试1: 检查categories表
            self.cur.execute("SELECT COUNT(*) FROM categories WHERE name LIKE 'cs.%';")
            total_cs_categories = self.cur.fetchone()[0]
            print(f"Total CS categories in database: {total_cs_categories}")
            
            # 测试2: 列出所有CS分类
            self.cur.execute("SELECT name FROM categories WHERE name LIKE 'cs.%' ORDER BY name;")
            all_cs = [row[0] for row in self.cur.fetchall()]
            print(f"All CS categories: {all_cs}")
            
            # 测试3: 检查已知分类是否存在
            existing_known = [cat for cat in all_cs if cat in self.known_categories]
            print(f"Known categories found in database: {existing_known}")
            
            # 测试4: 检查其他分类
            other_cs = [cat for cat in all_cs if cat not in self.known_categories]
            print(f"Other CS categories: {other_cs}")
            
            # 测试5: 简单计数测试
            if other_cs:
                test_category = other_cs[0]
                self.cur.execute("""
                    SELECT COUNT(*) FROM papers p
                    JOIN paper_category pc ON p.arxiv_id = pc.paper_arxiv_id
                    JOIN categories c ON pc.category_id = c.id
                    WHERE c.name = %s;
                """, (test_category,))
                test_count = self.cur.fetchone()[0]
                print(f"Test: Papers in '{test_category}': {test_count}")
            
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
        print("Database Analysis Results (Including Other CS Categories)")
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
    
    def save_to_csv(self, df, filename="./csv/database_analysis_with_others.csv"):
        """保存到CSV文件"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n✓ Results saved to {filename}")
    
    def run_analysis(self, mode="all"):
        """
        运行分析
        mode: "known" - 只查询已知分类
              "others" - 只查询其他cs分类  
              "all" - 查询所有分类（默认）
        """
        try:
            print(f"Starting database analysis (mode: {mode})...")
            
            # 根据模式执行不同的查询
            if mode == "known":
                results = self.execute_all_queries(include_known_categories=True, include_other_cs=False)
            elif mode == "others":
                results = self.execute_all_queries(include_known_categories=False, include_other_cs=True)
            else:  # mode == "all"
                results = self.execute_all_queries(include_known_categories=True, include_other_cs=True)
            
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
        
        # 选项1：只查询其他cs分类
        df = analyzer.run_analysis(mode="others")
        
        # 选项2：查询所有分类（已知的 + 其他的）
        # df = analyzer.run_analysis(mode="all")
        
        # 选项3：只查询已知分类
        # df = analyzer.run_analysis(mode="known")
    else:
        print("Database connection test failed. Please check your database setup.")