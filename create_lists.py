import os

def create_list_file(data_dir, output_file):
    image_dir = os.path.join(data_dir, 'images')
    gt_dir = os.path.join(data_dir, 'ground-truth')
    
    with open(output_file, 'w') as f:
        for img_file in sorted(os.listdir(image_dir)):
            if img_file.endswith('.jpg'):
                img_path = os.path.join('part_A', os.path.basename(data_dir), 'images', img_file)
                gt_file = f"GT_{img_file.replace('processed_', '').replace('.jpg', '.mat')}"
                gt_path = os.path.join('part_A', os.path.basename(data_dir), 'ground-truth', gt_file)
                
                full_gt_path = os.path.join(os.path.dirname(data_dir), os.path.basename(data_dir), 'ground-truth', gt_file)
                if os.path.exists(full_gt_path):
                    f.write(f"{img_path} {gt_path}\n")

# Criar listas para treinamento e teste
train_data_dir = './new_public_density_data/part_A/train_data'
test_data_dir = './new_public_density_data/part_A/test_data'

create_list_file(train_data_dir, './new_public_density_data/shanghai_tech_part_a_train.list')
create_list_file(test_data_dir, './new_public_density_data/shanghai_tech_part_a_test.list')

print("Arquivos de lista criados com sucesso!") 