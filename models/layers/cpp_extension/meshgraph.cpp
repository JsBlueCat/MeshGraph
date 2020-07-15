#include <torch/extension.h>

at::Tensor get_connect_matrix(const at::Tensor &faces, const int64_t &num_nodes)
{
    /*
    get connect matrix
    */
    std::vector<std::vector<int64_t>> node_list(num_nodes, std::vector<int64_t>(0));
#pragma omp parallel for
    for (int64_t i = 0; i < faces.size(0); i++)
    {
        int64_t *col = faces[i].data<int64_t>();
#pragma omp parallel for
        for (int j = 0; j < 3; j++)
        {
            node_list[col[j]].push_back(i);
        }
    }
    at::Tensor result = at::zeros(2, faces.options());
#pragma omp parallel for
    for (int64_t it = 0; it < num_nodes; it++)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < node_list[it].size() - 1; i++)
        {
#pragma omp parallel for
            for (int64_t j = i + 1; j < node_list[it].size(); j++)
            {
                std::cout << i << " " << j << std::endl;
                auto temp = at::empty(2, faces.options());
                temp[0] = i;
                temp[1] = j;
                result = at::cat({result, temp}, 0);
            }
        }
    }

    return result.view({-1, 2});
}

std::string test()
{
    std::cout << "hello world" << std::endl;
    return "hello_world";
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_connect_matrix", &get_connect_matrix, "get_connect_matrix");
    m.def("test", &test, "test function");
}
