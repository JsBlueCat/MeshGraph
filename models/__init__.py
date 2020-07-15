def create_model(opt):
    from .mesh_graph import mesh_graph
    model = mesh_graph(opt)
    return model
