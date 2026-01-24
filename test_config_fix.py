#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

try:
    from agi_persistent_evolution import PersistentAGIConfig
    print('✓ 导入配置成功')

    config = PersistentAGIConfig()
    print('✓ 配置初始化成功')

    # 测试TrainingArguments配置是否正确
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(config.checkpoint_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",       # 评估策略 (新版本transformers)
        save_strategy="steps",       # 保存策略，与评估策略匹配
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        fp16=config.use_mixed_precision,
        gradient_checkpointing=config.use_gradient_checkpointing,
        report_to="none"  # 禁用wandb以简化测试
    )
    print('✓ TrainingArguments配置成功')

    print('🎉 训练器配置修复验证完成！')

except Exception as e:
    print(f'❌ 配置测试失败: {e}')
    import traceback
    traceback.print_exc()