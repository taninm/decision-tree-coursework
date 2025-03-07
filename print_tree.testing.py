def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for feature, branches in tree.items():
            for value, subtree in branches.items():
                print(f"{indent}Feature: {feature}, Value: {value}")
                print_tree(subtree, indent + "  |-- ")
    else:
        print(f"{indent}Leaf: {tree}")


tree = {
    'persons': {
        2: {
            'safety': {
                'low': {
                    'buying': {
                        'vhigh': {
                            'maint': {
                                'vhigh': 'unacc'
                            }
                        }
                    }
                }
            }
        }
    }
}

print_tree(tree)

