#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

try:
    from agi_persistent_evolution import PersistentAGIConfig, PersistentAGITrainer
    print('✓ 导入成功')

    config = PersistentAGIConfig()
    print('✓ 配置初始化成功')

    trainer = PersistentAGITrainer(config)
    print('✓ 训练器初始化成功')

    # 测试模型初始化
    trainer.initialize_model()
    print('✓ 模型初始化成功')

    # 测试数据集设置
    train_dataset, eval_dataset, data_collator = trainer.setup_datasets()
    print(f'✓ 数据集设置成功 - 训练集: {len(train_dataset)} 条, 评估集: {len(eval_dataset)} 条')

    # 测试训练器设置
    trainer.setup_trainer(train_dataset, eval_dataset, data_collator)
    print('✓ 训练器设置成功')

    print('🎉 所有组件初始化成功！训练器配置修复完成！')

except Exception as e:
    print(f'❌ 初始化失败: {e}')
    import traceback
    traceback.print_exc()