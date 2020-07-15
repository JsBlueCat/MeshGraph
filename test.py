from options.test_options import test_options
from data.ModelNet import ModelNet
from preprocess.preprocess import FaceToGraph
from torch_geometric.data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = test_options().parse()
    dataset = ModelNet(root=opt.datasets, name='40_graph', train=False,
                       pre_transform=FaceToGraph(remove_faces=True))
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    model = create_model(opt)
    writer = Writer(opt)
    writer.reset_counter()
    for i, data in enumerate(loader):
        if data.y.size(0) % 64 != 0:
            continue
        model.set_input_data(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
