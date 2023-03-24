#!/usr/bin/python3
#
# Copyright (c) 2023 Oracle and/or its affiliates. All rights reserved.
# DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
#
# This code is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 only, as
# published by the Free Software Foundation.
#
# This code is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# version 2 for more details (a copy is included in the LICENSE file that
# accompanied this code).
#
# You should have received a copy of the GNU General Public License version
# 2 along with this work; if not, write to the Free Software Foundation,
# Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
# or visit www.oracle.com if you need additional information or have any
# questions.

import sys
import argparse
import xml.etree.cElementTree as et
import xml.dom.minidom as minidom
import networkx as nx
import os
import io
from pathlib import *
import shutil
import re
import random

# Helper functions for traversing the XML graph.

def find_node(graph, idx):
    for node in graph.find('nodes'):
        if int(node.attrib['id']) == idx:
            return node
    assert False

def find_node_properties(graph, idx):
    node = find_node(graph, idx)
    assert node != None
    ps = {}
    for p in node.find('properties'):
        ps[p.attrib['name']] = p.text.strip()
    return ps

def xml2graphs(xml_root, args):
    graphs = {}
    graph_id = 0
    for group in xml_root:
        method = group.find('method')
        group_name = method.attrib['name']
        bci = int(method.attrib['bci'])
        for graph in group.findall('graph'):
            graph_name = graph.attrib['name']
            if not matches((graph_id, group_name, graph_name), args.filter):
                graph_id += 1
                continue
            # Load the entire graph first.
            G = nx.MultiDiGraph()
            if not args.list:
                for node in graph.find('nodes'):
                    idx = int(node.attrib['id'])
                    properties = find_node_properties(graph, idx)
                    G.add_node(idx, **properties)
                for edge in graph.find('edges'):
                    src = int(edge.attrib['from'])
                    dst = int(edge.attrib['to'])
                    ind = int(edge.attrib['index'])
                    # The XML file sometimes contains (src,dst,ind) duplicates.
                    if not G.has_edge(src, dst, key=ind):
                        G.add_edge(src, dst, key=ind)
            # Load the control-flow graph, if available.
            CFG = None
            if graph.find('controlFlow'):
                CFG = nx.DiGraph()
                if not args.list:
                    for xmlblock in graph.find('controlFlow'):
                        block = int(xmlblock.attrib['name'])
                        # The node order reflects the local schedule.
                        nodes = []
                        for xmlnode in xmlblock.find('nodes'):
                            node = int(xmlnode.attrib['id'])
                            nodes.append(node)
                        CFG.add_node(block, **{'nodes' : nodes})
                        for xmlsucc in xmlblock.find('successors'):
                            succ = int(xmlsucc.attrib['name'])
                            CFG.add_edge(block, succ)
            graphs[graph_id] = ((group_name, graph_name), G, CFG)
            graph_id += 1
    return graphs

def graphs2xml(graphs, args):
    group_to_graphs = dict()
    for ((group, graph), G, CFG) in graphs.values():
        if not group in group_to_graphs:
            group_to_graphs[group] = []
        group_to_graphs[group].append((graph, G))

    xml_root = et.Element('graphDocument')
    for group, graphs in group_to_graphs.items():
        xml_group = et.SubElement(xml_root, 'group')
        xml_group_properties = et.SubElement(xml_group, 'properties')
        et.SubElement(xml_group_properties, 'p', name='name').text = group
        for (graph, G) in graphs:
            graph2xml(xml_group, graph, G, args)
    return xml_root

def graph2xml(xml_group, graph, G, args):
    xml_graph = et.SubElement(xml_group, 'graph', name=graph)
    xml_nodes = et.SubElement(xml_graph, 'nodes')
    for n in list(G.nodes()):
        xml_node = et.SubElement(xml_nodes, 'node', id=str(n))
        xml_node_properties = et.SubElement(xml_node, 'properties')
        for property, value in G.nodes[n].items():
            et.SubElement(xml_node_properties, 'p', name=property).text = value
    xml_edges = et.SubElement(xml_graph, 'edges')
    for e in list(G.edges):
        (src, dst, index) = e
        et.SubElement(xml_edges, 'edge', dict([('from', str(src)), ('to', str(dst)), ('index', str(index))]))
    et.SubElement(xml_graph, 'edges')

def expand(args, graph, key, expanded_graphs):
    n = int(args.size)
    ((method, phase), G, CFG) = graph
    for k in range(key, key + n):
        G = G.copy()
        step(args, G)
        expanded_graphs[k] = ((method, phase), G, CFG)
    return key + n

def step(args, G):
    # Simulate action on G (e.g. hiding or expansion).
    # Here, for example, we just remove one randomly picked node.
    nodes = list(G.nodes)
    if not nodes:
        return
    n = random.choice(nodes)
    G.remove_node(n)

filter_symbols = {'g' : 'int g',
                  'method' : 'str method(int)',
                  'phase' : 'str phase(int)'}
def matches(graph_tuple, filter):
    (g, m, p) = graph_tuple
    method = lambda g : m
    phase  = lambda g : p
    loc = locals()
    filter_locals = dict([(sym, loc[sym]) for sym in filter_symbols.keys()])
    return eval(filter, {}, filter_locals)

def add_feature_argument(parser, feature, help_msg, default):
    """
    Add a Boolean, mutually-exclusive feature argument to a parser.
    """
    if default:
        default_option = '--' + feature
    else:
        default_option = '--no-' + feature
    help_string = help_msg + " (default: " + default_option + ")"
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_lower = feature.replace('-', '_')
    feature_parser.add_argument('--' + feature,
                                dest=feature_lower,
                                action='store_true',
                                help=help_string)
    feature_parser.add_argument('--no-' + feature,
                                dest=feature_lower,
                                action='store_false',
                                help=argparse.SUPPRESS)
    parser.set_defaults(**{feature_lower:default})

def main():
    parser = argparse.ArgumentParser(
        description="Generates a sequence of graphs by simulating different user actions on a given graph.",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
        usage='%(prog)s [options] XML_FILE XML_OUTPUT_FILE')

    io = parser.add_argument_group('input/output options')
    io.add_argument('XML_INPUT_FILE',
                    help="XML input graph file emitted by the HotSpot JVM")
    io.add_argument('XML_OUTPUT_FILE',
                    help="XML output graph file")
    add_feature_argument(io,
                         'verbose',
                         "print debug information to the standard output",
                         False)
    io.add_argument('--help',
                    action='help',
                    default=argparse.SUPPRESS,
                    help='Show this help message and exit')
    list_filter = parser.add_argument_group('listing and filtering options')
    add_feature_argument(list_filter,
                         'list',
                         "list properties of each graph and terminate",
                         False)
    list_filter.add_argument('--filter',
                             metavar='EXP',
                             default='True',
                             help=
"""predicate telling whether to consider graph g (default: %(default)s)
-- arbitrary Python expression combining the following elements:
""" + '\n'.join(filter_symbols.values()))
    list_filter.add_argument('--size',
                             metavar='N',
                             default='1',
                             help="number of graph copies for each graph (default: %(default)s)")

    args = parser.parse_args()

    try:
        # Parse XML file.
        if args.verbose:
            print("parsing input file " + args.XML_INPUT_FILE + " ...")
        tree = et.parse(args.XML_INPUT_FILE)
        root = tree.getroot()

        # Convert XML to a map from id to ((method, phase), NetworkX graph,
        # maybe CFG) tuples.
        if args.verbose:
            print("converting XML to graphs ...")
        graphs = xml2graphs(root, args)

        # If asked for, list the graphs (id, method, phase).
        if args.verbose or args.list:
            table = [('id', 'method', 'phase')] + \
                [(graph_id, method, phase)
                 for (graph_id, ((method, phase), _, __)) in graphs.items()]
            ws = [max(map(len, map(str, c))) for c in zip(*table)]
            for r in table:
                print('  '.join((str(v).ljust(w) for v, w in zip(r, ws))))
        # If asked explicitly, terminate at this point.
        if args.list:
            return
        # Expand the selected graphs
        expanded_graphs = dict()
        key = 0
        for (graph_id, ((method, phase), G, CFG)) in graphs.items():
            if args.verbose:
                print("expanding " + str(graph_id) + " " + method + "::" + phase + "...")
            key = expand(args, ((method, phase), G, CFG), key, expanded_graphs)
        # Convert the selected graphs back to XML.
        if args.verbose:
            print("emitting output file " + args.XML_OUTPUT_FILE + " ...")
        out_root = graphs2xml(expanded_graphs, args)
        # Save output into file.
        xmlstr = minidom.parseString(et.tostring(out_root)).toprettyxml(indent="   ")
        with open(args.XML_OUTPUT_FILE, "w") as f:
            f.write(xmlstr)
    except Exception as error:
        print('Exception: {}'.format(error))
    finally:
        return

if __name__ == '__main__':
    main()
