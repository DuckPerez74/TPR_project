"""
Script provisório para extrair entidades já processadas dos modelos existentes
e criar o ficheiro de checkpoint.

Analisa os ficheiros em models/route_models/10min (última a ser criada por entidade)
e extrai as entidades.
"""

import os
from pathlib import Path

def extract_entities_from_models():
    # Pasta route_models/10min é a última a ser criada por entidade
    # Se existe ficheiro aqui, a entidade foi completamente processada
    models_base = Path(__file__).parent / 'models'
    route_10min = models_base / 'route_models' / '10min'
    
    entities = set()
    
    if not route_10min.exists():
        print(f"  Pasta não existe: {route_10min}")
        return entities
        
    files = list(route_10min.glob('*.joblib'))
    print(f"\n  Analisando {route_10min}: {len(files)} ficheiros")
    
    for file in files:
        filename = file.stem  # nome sem extensão
        
        # Formato: {entity}_{route}.joblib -> ex: 3122_v3_users.joblib
        # A entidade é a primeira parte antes do primeiro underscore
        parts = filename.split('_')
        if parts:
            entity = parts[0]
            if entity and len(entity) > 0:
                entities.add(entity)
    
    return entities


def main():
    print("=" * 60)
    print("Extração de Entidades Processadas")
    print("=" * 60)
    
    models_base = Path(__file__).parent / 'models'
    checkpoint_file = models_base / 'training_checkpoint.txt'
    
    print(f"\nCheckpoint file: {checkpoint_file}")
    
    # Verificar se já existe checkpoint
    existing_entities = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            existing_entities = set(line.strip() for line in f if line.strip())
        print(f"Checkpoint existente: {len(existing_entities)} entidades")
    
    # Extrair entidades dos modelos
    print("\nAnalisando modelos existentes...")
    new_entities = extract_entities_from_models()
    
    print(f"\n" + "=" * 60)
    print(f"Entidades encontradas nos modelos: {len(new_entities)}")
    
    # Juntar com existentes
    all_entities = existing_entities | new_entities
    added = len(all_entities) - len(existing_entities)
    
    print(f"Entidades no checkpoint anterior: {len(existing_entities)}")
    print(f"Novas entidades a adicionar: {added}")
    print(f"Total após merge: {len(all_entities)}")
    
    # Mostrar algumas entidades como exemplo
    sample = list(new_entities)[:10]
    print(f"\nExemplo de entidades encontradas: {sample}")
    
    # Confirmar antes de escrever
    if added > 0:
        response = input(f"\nEscrever {len(all_entities)} entidades no checkpoint? (s/n): ")
        if response.lower() == 's':
            # Criar pasta se não existir
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Escrever ficheiro
            with open(checkpoint_file, 'w') as f:
                for entity in sorted(all_entities):
                    f.write(f"{entity}\n")
            
            print(f"\n✓ Checkpoint guardado: {checkpoint_file}")
            print(f"✓ Total de {len(all_entities)} entidades")
        else:
            print("\nOperação cancelada.")
    else:
        print("\nNenhuma nova entidade para adicionar.")


if __name__ == '__main__':
    main()
