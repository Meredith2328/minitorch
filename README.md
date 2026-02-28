# minitorch
The full minitorch student suite. 


To access the autograder: 

* Module 0: https://classroom.github.com/a/qDYKZff9
* Module 1: https://classroom.github.com/a/6TiImUiy
* Module 2: https://classroom.github.com/a/0ZHJeTA0
* Module 3: https://classroom.github.com/a/U5CMJec1
* Module 4: https://classroom.github.com/a/04QA6HZK
* Quizzes: https://classroom.github.com/a/bGcGc12k

## 补：环境配置说明

官方requirements.txt没有固定一些包的版本，所以导致出现包版本不匹配的问题。我根据自己成功跑通的版本，更新了requirements.txt。按照以下步骤即可：

```bash
conda create -n myminitorch python==3.11
pip install -r requirements.txt
pip install -Ue .
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

上述步骤共计4步，conda创建环境、安装前置包、在开发模式安装当前minitorch包、根据CUDA安装相应版本的torch以便使用GPU（参照PyTorch官网教程）。

****

以下无用，只是根据官方requirements探索出可用的requirements的过程纪实。

官方setup给的requirements有conflict，参考了pr里的requirements
pip install -r requirements.txt
pip install -Ue .
之后发现torch版本很新、自动装的2.0的numpy不兼容，手动降级一下：
pip install numpy\==1.24
以及跑Task0.5出了一些id重复的报错，判断是Streamlit版本新了一点加了检查，为了省去改代码的麻烦手动降级一下：
pip install streamlit\==1.26.0

之后做到Efficiency(Task3那些)要用CUDA，我本地机CUDA==12.6，所以参考了pytorch官网的指令在环境里装gpu的pytorch：
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
