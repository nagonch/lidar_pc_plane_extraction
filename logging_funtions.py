def log_args(args):
    args_items = vars(args)
    log_string = "\n***BENCHMARK OPTIONS***\n\n"
    args = []
    values = []
    for arg, value in sorted(args_items.items()):
        args.append(arg)
        values.append(value)

    max_arg_len = max([len(x) for x in args])
    for arg, value in zip(args, values):
        spaces = ' ' * (max_arg_len-len(arg))
        log_string += f'\t{arg}:{spaces}    {value}\n'
    print(log_string)