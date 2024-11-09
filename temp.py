
def drop_edges(edge_index, drop_rate=0.1):
    """随机丢弃边，依据设定的丢弃比例。
    Args:
        edge_index (torch.Tensor): 边的张量，形状为 (2, num_edges)。
        drop_rate (float): 要丢弃的边的比例。
    Returns:
        torch.Tensor: 丢弃部分边后的 edge_index。
    """
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, mask]


for epoch in range(num_epoch):
    train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
    model.train()
    running_loss = 0.0
    running_loss1 = 0.0
    running_attention_loss = 0.0  # 记录 attention 正则化损失
    running_sparse_loss = 0.0  # 记录稀疏损失
    running_cell_loss = 0.0  # 记录 loss_cell
    

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", unit="batch")
    
    for idx, sample in enumerate(progress_bar):
        gene_num = sample.y_exp.shape[0]
        optimizer.zero_grad()
        edge_index_dropped = sample.edge_index
        gene_pre, atten, cell_pre = model(
            sample.seq_data.to(device),
            sample.x.to(device),
            sample.edge_index.to(device),
            sample.edge_tf.T.to(device),
            sample.batch.to(device), 
            gene_num, 
            sample.id_vec.to(device)
        )
        edge_temp = model.edge
        index = torch.where(sample.x[sample.id_vec == 1] > 0)[0]
        loss1 = -loss_exp(gene_pre.flatten()[index], sample.y_exp.to(device)[index]) 
        loss_cell = criterion2(cell_pre, sample.y.to(device))
        temp_var_edge = torch.stack(torch.split(edge_temp, sample.edge_index.shape[1] // len(sample.y)))
        temp_var_atten = torch.stack(torch.split(atten[1], sample.edge_index.shape[1] // len(sample.y)))
        attention_loss = attention_reg_weight * criterion_sparse2(temp_var_atten)
        
        loss2 = sparse_loss_weight * criterion_sparse1(temp_var_edge)
        
        loss = loss1 + attention_loss + loss2 + loss_cell 
  
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        running_loss += loss.item()
        running_loss1 += loss1.item()
        running_attention_loss += attention_loss.item()  # 记录注意力正则化损失
        running_sparse_loss += loss2.item()  # 记录稀疏损失
        running_cell_loss += loss_cell.item()  # 记录 loss_cell
        
        if idx % 200 == 0:
            print(gene_pre, edge_temp, atten)
            print(f"gene_pre: {gene_pre}, loss_cell: {loss_cell.item()}, attention_loss: {attention_loss.item()}, sparse_loss: {loss2.item()}")
        
        progress_bar.set_postfix(
            loss=running_loss / (progress_bar.n + 1),
            loss1=running_loss1 / (progress_bar.n + 1),
            attention_loss=running_attention_loss / (progress_bar.n + 1),
            sparse_loss=running_sparse_loss / (progress_bar.n + 1),
            cell_loss=running_cell_loss / (progress_bar.n + 1)
        )
        
        torch.cuda.empty_cache()
    
    print(f"Epoch [{epoch+1}/{num_epoch}], Average Loss: {running_loss / len(train_loader):.4f}, "
          f"Loss1: {running_loss1 / len(train_loader):.4f}, Attention Loss: {running_attention_loss / len(train_loader):.4f}, "
          f"Sparse Loss: {running_sparse_loss / len(train_loader):.4f}, Cell Loss: {running_cell_loss / len(train_loader):.4f}")
