# 高通神经网络量化白皮书

- 神经网络量化会将权重和激活值张量以低bit存储。当从32bit存储变成8bit存储时，存储张量的内存开销变为原来的1/4，而矩阵乘法的计算开销变为原来的1/16。

## 硬件层面的背景

- 我们首先来考察下神经网络加速器里的矩阵乘以向量是怎么做的，如下图所示。为了提升推理的效率，这些硬件单元都是并行做计算。神经网络加速器的两个基本组件是处理单元$C_{n,m}$和加法器$A_n$。计算执行的顺序是：
  1. 把bias $\bold{b}_n$加载到加法器中；
  2. 把weight $\bold{W}_{n,m}$和输入的$\bold{x}_{m}$加载进数组，然后在一个时钟周期内在相应的处理单元$C_{n,m}=\bold{W}_{n,m}\bold{x}_{m}$计算它们的乘积。
  3. 结果在加法器中按照下式相加。
$$A_n=\bold{b}_n+\sum_{m}C_{n,m}$$
以上的操作就是一个**MAC**（*Multiply-Accumulate*）。对于大型矩阵向量乘法，这一步会重复很多次。
![alt text](1733354991410.png)
- 一旦所有的循环都完成了，累加器中的值就会被移回内存中，用于下一个神经网络层。
- 神经网络通常使用FP32的权值和激活值来进行训练。如果我们要在FP32中执行推理，处理元素和累加器将必须支持浮点逻辑，并且我们将需要将32位数据从内存传输到处理单元。
- **MAC操作和数据传输消耗了神经网络推理过程中所花费的大部分能量**。因此，通过对这些量使用较低的比特定点或量化表示，可以获得显著的好处。低位定点表示，如INT8，不仅减少了数据传输量，而且还减少了MAC操作的大小和能量消耗。乘法的成本通常与使用的bit位数呈二次关系，并且定点加法比浮点加法更高效。
- 通过量化权重和激活，我们可以写出累加的量化版本：

$$\begin{aligned}
\hat{A}_n & =\widehat{\mathbf{b}}_n+\sum_m \widehat{\mathbf{W}}_{n, m} \widehat{\mathbf{x}}_m \\
& =\widehat{\mathbf{b}}_n+\sum_m\left(s_{\mathbf{w}} \mathbf{W}_{n, m}^{\mathrm{int}}\right)\left(s_{\mathbf{x}} \mathbf{x}_m^{\mathrm{int}}\right) \\
& =\widehat{\mathbf{b}}_n+s_{\mathbf{w}} s_{\mathbf{x}} \sum_m \mathbf{W}_{n, m}^{\mathrm{int}} \mathbf{x}_m^{\mathrm{int}}
\end{aligned}$$

-
