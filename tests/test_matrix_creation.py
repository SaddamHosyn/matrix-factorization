from utils.matrix_creation import create_and_save_user_item_matrix

if __name__ == "__main__":
    train_df, test_df, norm_matrix, user_means = create_and_save_user_item_matrix()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("User-item matrix shape:", norm_matrix.shape)
    print("Sample of normalized matrix:")
    print(norm_matrix.head())
