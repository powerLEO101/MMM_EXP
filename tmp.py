# ---

ddp_world_size = 1
dataloader = MyDataLoader(train_df_path='test.csv',
                          misconceptions_path='misconception_mapping.csv',
                          batch_size=16, 
                          model_name='Salesforce/SFR-Embedding-Mistral',
                          rank=0,
                          folds=[0, 1, 2, 3, 4])
for one_batch in dataloader.all_text():
    print(one_batch)
    break

# ---
