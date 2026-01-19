#!/usr/bin/env python3
"""
Comprehensive architecture analysis:
1. Dependency graph of all modules
2. Module function categorization
3. Quaternion/Fractal architecture components identification
4. Data flow and architecture bottleneck analysis
"""
import os
import ast
import json
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict

class ArchitectureAnalyzer:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.modules = {}  # {module_path: {name, type, imports, size}}
        self.dependencies = defaultdict(set)  # {module -> set of dependencies}
        self.quaternion_modules = []
        self.fractal_modules = []
        self.acceleration_modules = []
        self.memory_modules = []
        self.inference_modules = []
        
    def scan_modules(self):
        """Scan all .py files and extract metadata."""
        print("[*] Scanning modules...")
        count = 0
        for py_file in self.root.rglob("*.py"):
            if any(x in str(py_file) for x in ['__pycache__', 'venv', '.git', 'temp_sandbox']):
                continue
            try:
                rel_path = str(py_file.relative_to(self.root))
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                tree = ast.parse(content)
                
                # Extract imports
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                
                # Extract classes/functions
                symbols = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        symbols.append({'name': node.name, 'type': 'class'})
                    elif isinstance(node, ast.FunctionDef):
                        symbols.append({'name': node.name, 'type': 'function'})
                
                lines = len(content.split('\n'))
                self.modules[rel_path] = {
                    'imports': imports,
                    'symbols': symbols,
                    'lines': lines,
                    'content': content[:500]  # snippet for pattern matching
                }
                self.dependencies[rel_path] = imports
                
                # Categorize by keyword patterns
                filename_lower = py_file.name.lower()
                content_lower = content.lower()
                
                if any(x in filename_lower or x in content_lower for x in ['quaternion', 'quat', 'q8']):
                    self.quaternion_modules.append(rel_path)
                if any(x in filename_lower or x in content_lower for x in ['fractal', 'recursive', 'tree']):
                    self.fractal_modules.append(rel_path)
                if any(x in filename_lower or x in content_lower for x in ['accelerat', 'amx', 'jit', 'metal', 'gpu']):
                    self.acceleration_modules.append(rel_path)
                if any(x in filename_lower or x in content_lower for x in ['memory', 'cache', 'paging', 'swap']):
                    self.memory_modules.append(rel_path)
                if any(x in filename_lower or x in content_lower for x in ['inference', 'beam', 'search', 'brain']):
                    self.inference_modules.append(rel_path)
                
                count += 1
            except Exception as e:
                print(f"  Warning: {py_file} - {e}")
        
        print(f"[+] Scanned {count} modules")
    
    def build_dependency_tree(self):
        """Build a dependency tree showing which modules depend on which."""
        print("[*] Building dependency tree...")
        tree = {}
        
        for module, imports in self.dependencies.items():
            if 'h2q' in imports or any(x in str(module) for x in self.modules.keys() if '/' in x):
                tree[module] = []
                for imp in imports:
                    # Simple matching: find modules that provide this import
                    for mod_path in self.modules.keys():
                        if imp in mod_path or mod_path.endswith(f'{imp}.py'):
                            tree[module].append(mod_path)
        
        return tree
    
    def compute_stats(self):
        """Compute architecture statistics."""
        total_lines = sum(m['lines'] for m in self.modules.values())
        total_modules = len(self.modules)
        
        stats = {
            'total_modules': total_modules,
            'total_lines': total_lines,
            'avg_lines_per_module': total_lines // max(1, total_modules),
            'quaternion_modules': len(self.quaternion_modules),
            'fractal_modules': len(self.fractal_modules),
            'acceleration_modules': len(self.acceleration_modules),
            'memory_modules': len(self.memory_modules),
            'inference_modules': len(self.inference_modules),
        }
        return stats
    
    def generate_report(self):
        """Generate comprehensive architecture report."""
        self.scan_modules()
        tree = self.build_dependency_tree()
        stats = self.compute_stats()
        
        report = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'statistics': stats,
            'quaternion_modules': self.quaternion_modules[:10],
            'fractal_modules': self.fractal_modules[:10],
            'acceleration_modules': self.acceleration_modules[:10],
            'memory_modules': self.memory_modules[:10],
            'inference_modules': self.inference_modules[:10],
            'top_modules_by_lines': sorted(
                [(p, self.modules[p]['lines']) for p in self.modules.keys()],
                key=lambda x: x[1],
                reverse=True
            )[:20],
            'core_imports': dict(sorted(
                [(imp, len([m for m, imps in self.dependencies.items() if imp in imps]))
                 for imp in set(imp for imps in self.dependencies.values() for imp in imps)],
                key=lambda x: x[1],
                reverse=True
            )[:20]),
        }
        
        return report, tree

if __name__ == '__main__':
    analyzer = ArchitectureAnalyzer('./h2q_project')
    report, tree = analyzer.generate_report()
    
    # Save detailed report
    with open('architecture_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("ARCHITECTURE ANALYSIS REPORT")
    print("="*60)
    print(f"\nStatistics:")
    for k, v in report['statistics'].items():
        print(f"  {k}: {v}")
    
    print(f"\n[Quaternion Modules] ({report['statistics']['quaternion_modules']} found):")
    for m in report['quaternion_modules'][:5]:
        print(f"  - {m}")
    
    print(f"\n[Fractal/Recursive Modules] ({report['statistics']['fractal_modules']} found):")
    for m in report['fractal_modules'][:5]:
        print(f"  - {m}")
    
    print(f"\n[Acceleration Modules] ({report['statistics']['acceleration_modules']} found):")
    for m in report['acceleration_modules'][:5]:
        print(f"  - {m}")
    
    print(f"\n[Memory Management Modules] ({report['statistics']['memory_modules']} found):")
    for m in report['memory_modules'][:5]:
        print(f"  - {m}")
    
    print(f"\n[Inference Modules] ({report['statistics']['inference_modules']} found):")
    for m in report['inference_modules'][:5]:
        print(f"  - {m}")
    
    print(f"\nTop imports/dependencies (core infrastructure):")
    for imp, count in list(report['core_imports'].items())[:15]:
        print(f"  {imp}: used by {count} modules")
    
    print("\nReport saved to: architecture_report.json")
