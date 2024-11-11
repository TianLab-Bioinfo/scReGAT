### Latest: <span style="color:red">请参考Cortex_fig.ipynb</span>
### 该版本无需DNABERT-2，支持批次训练
### 也没有细胞类别损失

### Welcome to scReGAT

#### Steps to use scReGAT:

1. **Download DNABERT-2**:
   - Visit: [DNABERT-2 117M](https://huggingface.co/zhihan1996/DNABERT-2-117M/tree/main)
   - Download all files and place them in the same folder on your GPU server.
   - Install the `transformers` package in your Python environment.

2. **Construct the graph**.

3. **Obtain sequences for each peak**.

4. **Generate sequence embeddings using DNABERT-2**.

5. **Run the model**:
   - Recommended loss function: KL Divergence.
   - Alternative loss function: MES loss.

6. **Score the binding intensity** based on attention scores and promoter activity.

---

**Summary**:

Embark on your journey with scReGAT by initializing DNABERT-2. Begin by constructing the graph and obtaining the necessary sequences for each peak. Use DNABERT-2 to generate sequence embeddings. When running the model, choose between KL Divergence (recommended) or MES loss for optimal performance. Finally, evaluate the binding intensity based on attention scores and promoter activity.

Happy coding! 🚀
