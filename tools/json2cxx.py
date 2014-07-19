#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# convert gbdt trees to a c++ function
#
# author: yafei(zhangyafeikimi@gmail.com)
#
import sys
import json
from optparse import OptionParser


DEFAULT_INPUT = '../output.json'
DEFAULT_OUTPUT = 'output.cc'
DEFAULT_CXX_NAMESPACE = 'gbdt'
INDENT = '    '
func_signature = ''


def get_header(ofile):
    cc_list = ['.cc', '.cxx', '.cpp']
    for cc in cc_list:
        if ofile.endswith(cc):
            return ofile.replace(cc, '.h')
    return ofile + '.h'


def generate_ns_begin(fp, namespace):
    if len(namespace) != 0:
        ns_list = namespace.split('::')
        for ns in ns_list:
            fp.write('namespace %s { ' % ns)
        fp.write('\n')


def generate_ns_end(fp, namespace):
    if len(namespace) != 0:
        ns_list = namespace.split('::')
        for ns in ns_list:
            fp.write('} ')
        fp.write('\n')


def generate_sig(trees):
    global func_signature
    func_signature = 'double predict(\n'
    spec = trees['spec']
    assert(len(spec) != 0)
    for i in range(0, len(spec)):
        s = spec[i]
        if s == 'numerical':
            func_signature += '%sdouble _arg%d,\n' % (INDENT, i)
        else:
            func_signature += '%sint _arg%d,\n' % (INDENT, i)
    func_signature = func_signature[0:-2] + ')'


def generate_indent(fp, indent):
    for i in range(0, indent):
        fp.write(INDENT)
    return fp


def _generate_tree(fp, node, indent):
    if node.has_key('split_index'):
        index = node['split_index']

        generate_indent(fp, indent).write('if (_arg%d <= %s)\n' % (index, node['split_value']))
        generate_indent(fp, indent).write('{\n')
        _generate_tree(fp, node['left'], indent+1)
        generate_indent(fp, indent).write('}\n')

        generate_indent(fp, indent).write('else\n')
        generate_indent(fp, indent).write('{\n')
        _generate_tree(fp, node['right'], indent+1)
        generate_indent(fp, indent).write('}\n')
    else:
        generate_indent(fp, indent).write('y += %f;\n' % node['value'])


def generate_tree(fp, root):
    _generate_tree(fp, root, 1)


def generate_h(fp, trees, namespace):
    guard_macro = ''
    fname = fp.name.upper()
    for c in fname:
        if c.isalpha() or c.isdigit():
            guard_macro = guard_macro + c
        else:
            guard_macro = guard_macro + '_'
    fp.write('#ifndef %s\n' % guard_macro)
    fp.write('#define %s\n' % guard_macro)
    fp.write('\n')
    generate_ns_begin(fp, namespace)
    fp.write('\n')

    fp.write(func_signature)
    fp.write(';\n')

    fp.write('\n')
    generate_ns_end(fp, namespace)
    fp.write('\n')
    fp.write('#endif // %s\n' % guard_macro)


def generate_cc(fp, trees, namespace):
    generate_ns_begin(fp, namespace)
    fp.write('\n')

    fp.write(func_signature)
    fp.write('\n')
    fp.write('{\n')
    fp.write('%sdouble y = %f;\n' % (INDENT, trees['y0']))
    for root in trees['trees']:
        generate_tree(fp, root)
    fp.write('%sreturn y;\n' % INDENT)
    fp.write('}\n')

    fp.write('\n')
    generate_ns_end(fp, namespace)


def main(ifile, ofile, namespace):
    print 'loading %s' % ifile
    fp = open(ifile, 'r')
    trees = json.load(fp)
    fp.close()

    generate_sig(trees)

    print 'writing %s' % ofile
    fp = open(ofile, 'w')
    generate_cc(fp, trees, namespace)
    fp.close()

    print 'writing %s' % get_header(ofile)
    fp = open(get_header(ofile), 'w')
    generate_h(fp, trees, namespace)
    fp.close()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='ifile',
            default=DEFAULT_INPUT,
            help='input file, default is "%s"' % DEFAULT_INPUT)
    parser.add_option('-o', '--output', dest='ofile',
            default=DEFAULT_OUTPUT,
            help='output file, default is "%s"' % DEFAULT_OUTPUT)
    parser.add_option('-c', '--cxx-namespace', dest='namespace',
            default=DEFAULT_CXX_NAMESPACE,
            help='c++ namespace, default is "%s"' % DEFAULT_CXX_NAMESPACE)
    (options, args) = parser.parse_args()
    main(options.ifile, options.ofile, options.namespace)
