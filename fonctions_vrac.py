def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, (type, ModuleType, FunctionType)):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, (type, ModuleType, FunctionType)) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


# Densité d'une matrice

sparsity = 1.0 - (np.count_nonzero(simatrix) / float(simatrix.size))
density = 1 - sparsity
print(sparsity)
print(density)