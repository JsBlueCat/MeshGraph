# MeshGraph in PyTorch

Transfroming Mesh data to Mesh Graph Tology through the idea of Finite Element, Paper is publishing soon.
![transfrom](img/1.png)
![network](img/3.png)
# Getting Started 

### Installation
- Clone this repo:
``` bash 
git clone https://github.com/JsBlueCat/MeshGraph.git
cd MeshVertexNet
```
- Install dependencies: [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and [docker](https://docs.docker.com/get-started/)

- First intall docker image

```bash
cd docker
docker build -t your/docker:meshgraph.
```

- then run docker image
```bash
docker run --rm -it --runtime=nvidia --shm-size 16G -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /your/path/to/MeshGraph/:/meshgraph your/docker:meshgraph bash
```


### 3D Shape Classification on ModelNet40
- get the dataset if you fail download it from python [[Model40]](https://drive.google.com/uc?export=download&confirm=HB4c&id=1o9pyskkKMxuomI5BWuLjCG2nSv5iePZz)
- put the zip file in datasets/modelnet40_graph

```bash 
cd /meshvertex
sh script/modelnet40_graph/train.sh 
```


### Classification Acc. On ModelNet40
| Method         | Modality | Acc |
| ----------------------- |:--------:|:--------:|
| 3DShapeNets | volume | 77.3% |
| Voxnet | volume | 83% |
| O-cnn | volume | 89.9% |
| Mvcnn | view | 90.1% |
| Mlvcnn | view | 94.16% |
| Pointnet | Point cloud | 89.2% |
| Meshnet | Mesh | 91% |
| Ours with SAGE | Mesh | 94.3% +0.5% |
- run test
- get the dataset if you fail download it from python [[Model40]](https://drive.google.com/uc?export=download&confirm=HB4c&id=1o9pyskkKMxuomI5BWuLjCG2nSv5iePZz)
- put the zip file in datasets/modelnet40_graph
- download the weight file from [[weights]](https://drive.google.com/file/d/11JOiaTOBCykCYgZKw24qcD6r1Tzz7dvu/view?usp=sharing)  and put it in your ckpt_root/40_graph/ and run 
``` bash
sh script/modelnet40_graph/test.sh 
```
- the result will be like 
``` bash 
root@2730e382330f:/meshvertex# sh script/modelnet40_graph/test.sh 
Running Test
loading the model from ./ckpt_root/40_graph/final_net.pth
epoch: -1, TEST ACC: [94.49 %]
```
![result](img/2.png)

# Train on your Dataset
### Coming soon

# Credit

### MeshGraphNet: An Effective 3D Polygon Mesh Recognition with Topology Reconstruction
AnPing S,XinYi D <br>

**Abstract** <br>
Three-dimensional polygon mesh recognition has a significant impact on current computer graphics. However, its application to some real-life fields, such as unmanned driving and medical image processing, has been restricted due to the lack of inner-interactivity, shift-invariance, and  numerical uncertainty of mesh surfaces. In this paper, an interconnected topological dual graph that extracts adjacent information from each triangular face of a polygon mesh is constructed, in order to address the above issues. On the basis of the algebraic topological graph, we propose a mesh graph neural network, called MeshGraphNet, to effectively extract features from mesh data. In this method, the graph node-unit and correlation between every two dual graph vertexes are defined, and the concept of aggregating features extracted from geodesically adjacent nodes is introduced. A graph neural network with available and effective blocks is proposed. With these methods, MeshGraphNet performs well in 3D shape representation, while avoiding the lack of inner-interactivity, shift-invariance, and the numerical uncertainty problems of mesh data. We conduct extensive 3D shape classification experiments and provide visualizations of the features extracted from the fully connected layers. The results demonstrate that our method performs better than state-of-the-art methods and improves the recognition accuracy by 4--4.5%.