# ===== 1. DATASET & UTILS =====
class ARCDataset(Dataset):
    def __init__(self, challenges_path: str, solutions_path: str = None, augment: bool = True):
        self.augment = augment
        self.examples = []
        
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)
        
        solutions = {}
        if solutions_path and os.path.exists(solutions_path):
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)

        for task_id, task in challenges.items():
            # Training pairs in the task
            for ex in task['train']:
                self.examples.append({
                    'input': torch.tensor(ex['input'], dtype=torch.long),
                    'output': torch.tensor(ex['output'], dtype=torch.long)
                })
            # Test pairs (if solutions are provided)
            if task_id in solutions:
                for i, ex in enumerate(task['test']):
                    sol_out = solutions[task_id][i]
                    output_grid = sol_out['output'] if isinstance(sol_out, dict) else sol_out
                    self.examples.append({
                        'input': torch.tensor(ex['input'], dtype=torch.long),
                        'output': torch.tensor(output_grid, dtype=torch.long)
                    })

    def __len__(self):
        return len(self.examples)

    def _apply_augmentation(self, inp, out):
        # 1. Random Rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        inp = torch.rot90(inp, k, [0, 1])
        out = torch.rot90(out, k, [0, 1])

        # 2. Random Flips
        if random.random() > 0.5:
            inp = torch.flip(inp, [0]) # Vertical
            out = torch.flip(out, [0])
        if random.random() > 0.5:
            inp = torch.flip(inp, [1]) # Horizontal
            out = torch.flip(out, [1])

        # 3. Color Permutation (The "Logic Injection")
        if random.random() > 0.5:
            # Create a mapping for colors 0-9
            colors = torch.randperm(10)
            # Add index 10 for padding (must remain 10)
            mapping = torch.cat([colors, torch.tensor([10])])
            
            inp = mapping[inp]
            out = mapping[out]
            
        return inp, out

    def __getitem__(self, idx):
        item = self.examples[idx]
        inp = item['input'].clone()
        out = item['output'].clone()

        if self.augment:
            inp, out = self._apply_augmentation(inp, out)

        return {'input': inp, 'output': out}

# ===== 2. COLLATE FUNCTION (UNCHANGED STRUCTURE) =====
def collate_fn(batch):
    B = len(batch)
    inputs = torch.full((B, 30, 30), 10, dtype=torch.long) # 10 is padding color
    outputs = torch.full((B, 30, 30), 10, dtype=torch.long)
    target_sizes = torch.zeros((B, 2), dtype=torch.long)
    
    for i, item in enumerate(batch):
        hi, wi = item['input'].shape
        inputs[i, :hi, :wi] = item['input']
        ho, wo = item['output'].shape
        outputs[i, :ho, :wo] = item['output']
        target_sizes[i] = torch.tensor([ho, wo])
        
    return {'input': inputs, 'output': outputs, 'target_size': target_sizes}
