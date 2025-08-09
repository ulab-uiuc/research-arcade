# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# from paper_crawler.task_database import TaskDatabase


# tdb = TaskDatabase()

# time1 = "2025-02-01"
# time2 = "2025-02-22"
# time3 = "2025-03-01"
# time4 = "2025-06-01"
# time5 = "2025-02-01"
# time6 = "2025-07-01"

# tdb.create_paper_search_intervals_table()
# interval = tdb.insert_paper_search_intervals(time1, time2, "done")
# print(interval)
# interval = tdb.insert_paper_search_intervals(time3, time4, "done")
# print(interval)
# interval = tdb.insert_paper_search_intervals(time5, time6, "done")
# print(interval)
# tdb.drop_paper_search_intervals_table()