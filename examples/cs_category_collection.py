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
        
        # 定义分类和查询
        self.categories = ["cs.AI", "cs.LG", "cs.CV", "cs.RO", "cs.CR", 
                          "cs.DB", "cs.DC", "cs.PF", "cs.MA", "cs.OS"]
        
        self.statements = [
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
        
        # 列名
        self.column_names = ["paper", "section", "paragraph", "figure", "table", "author"]
    
    def execute_queries(self):
        """执行所有查询并返回结果矩阵"""
        results = []
        
        for category in self.categories:
            row = [category]  # 第一列是分类名
            
            for statement in self.statements:
                try:
                    self.cur.execute(statement, (category,))
                    count = self.cur.fetchone()[0]
                    row.append(count)
                    print(f"✓ {category} - {self.column_names[len(row)-2]}: {count}")
                except Exception as e:
                    print(f"✗ Error for {category}: {e}")
                    row.append(0)  # 出错时填0
            
            results.append(row)
        
        return results
    
    def create_dataframe(self, results):
        """创建pandas DataFrame"""
        columns = ["category"] + self.column_names
        df = pd.DataFrame(results, columns=columns)
        return df
    
    def print_table(self, df):
        """打印格式化的表格"""
        print("\n" + "="*80)
        print("Database Analysis Results")
        print("="*80)
        
        # 打印表头
        header = f"{'Category':<10}"
        for col in self.column_names:
            header += f"{col.capitalize():<12}"
        print(header)
        print("-" * 80)
        
        # 打印数据行
        for _, row in df.iterrows():
            line = f"{row['category']:<10}"
            for col in self.column_names:
                line += f"{row[col]:<12}"
            print(line)
    
    def save_to_csv(self, df, filename="./csv/database_analysis.csv"):
        """保存到CSV文件"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n✓ Results saved to {filename}")
    
    def run_analysis(self):
        """运行完整的分析流程"""
        try:
            print("Starting database analysis...")
            
            # 执行查询
            results = self.execute_queries()
            
            # 创建DataFrame
            df = self.create_dataframe(results)
            
            # 打印表格
            self.print_table(df)
            
            # 保存到CSV
            self.save_to_csv(df)
            
            # 显示一些统计信息
            print(f"\n📊 Summary:")
            print(f"Total categories analyzed: {len(self.categories)}")
            print(f"Total papers across all categories: {df['paper'].sum():,}")
            print(f"Total authors across all categories: {df['author'].sum():,}")
            
            return df
            
        except Exception as e:
            print(f"Error during analysis: {e}")
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
    df = analyzer.run_analysis()
    
    # # 如果需要进一步处理数据
    # if df is not None:
    #     # 例如：找到论文数量最多的分类
    #     max_papers = df.loc[df['paper'].idxmax()]
    #     print(f"\n🏆 Category with most papers: {max_papers['category']} ({max_papers['paper']} papers)")