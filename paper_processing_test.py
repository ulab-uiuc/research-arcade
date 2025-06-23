from graph_constructor.node_processor import NodeConstructor


nc = NodeConstructor()

nc.drop_tables()

nc.create_tables()
arxiv_id = "2501.02725"
dir_path = "download"
nc.process_paper(arxiv_id=arxiv_id, dir_path=dir_path)


