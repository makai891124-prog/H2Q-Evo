    def get_checkpoint_stats(self):
        """获取断点统计信息"""
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            print("❌ 没有找到任何断点文件")
            return

        print(f"📊 断点统计: 共 {len(checkpoints)} 个断点")
        print("-" * 50)

        for name, path in checkpoints:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                step = data.get('current_step', 0)
                loss = data.get('best_loss', 0)
                duration = data.get('training_duration', 'N/A')

                print("12s"
            except Exception as e:
                print("12s"