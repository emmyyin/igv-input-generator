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
import itertools

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

def find_name_in_properties(xmlelem):
    for p in xmlelem.find('properties'):
        if (p.attrib['name'] == 'name'):
            return p.text.strip()
    return None

def parse_add_node(G, xmlnode, xmlgraph):
    idx = int(xmlnode.attrib['id'])
    properties = find_node_properties(xmlgraph, idx)
    G.add_node(idx, **properties)

def parse_edge(G, xmledge):
    src = int(xmledge.attrib['from'])
    dst = int(xmledge.attrib['to'])
    if 'index' in xmledge.attrib:
        ind = int(xmledge.attrib['index'])
    elif 'toIndex' in xmledge.attrib:
        ind = int(xmledge.attrib['toIndex'])
    else:
        ind = 0
    return (src, dst, ind)

def parse_add_edge(G, xmledge):
    (src, dst, ind) = parse_edge(G, xmledge)
    # The XML file sometimes contains (src,dst,ind) duplicates.
    if not G.has_edge(src, dst, key=ind):
        G.add_edge(src, dst, key=ind)

def xml2graphs(xml_root, args):
    graphs = {}
    graph_id = 0
    for group in xml_root:
        is_difference = False
        if 'difference' in group.attrib:
            is_difference = group.attrib['difference'] == 'true'
        previousG = None
        group_name = find_name_in_properties(group)
        if args.verbose:
            print("  building " + group_name + " (difference: " + str(is_difference) + ")...")
        for graph in group.findall('graph'):
            if 'name' in graph.attrib:
                graph_name = graph.attrib['name']
            else:
                graph_name = find_name_in_properties(graph)
            # If the graph does not match the given filter pattern, skip parsing
            # and building it. Except if we are difference mode, where have to
            # parse and build the graph anyway to set it as the previous graph.
            if not is_difference and \
               not matches((graph_id, group_name, graph_name), args.filter):
                graph_id += 1
                continue
            if is_difference and previousG != None:
                # Incremental mode. Build the graph based on the previous one.
                if args.verbose:
                    print("    building " + graph_name + " incrementally...")
                G = previousG.copy()
                for node in graph.find('nodes'):
                    if node.tag == 'node': # Node insertion.
                        parse_add_node(G, node, graph)
                    elif node.tag == 'removeNode': # Node removal.
                        idx = int(node.attrib['id'])
                        G.remove_node(idx)
                    else:
                        assert False
                for edge in graph.find('edges'):
                    if edge.tag == 'edge': # Edge insertion.
                        parse_add_edge(G, edge)
                    elif edge.tag == 'removeEdge': # Edge removal.
                        (src, dst, ind) = parse_edge(G, edge)
                        # The edge may have been removed with one of its nodes.
                        if G.has_edge(src, dst, key=ind):
                            G.remove_edge(src, dst, key=ind)
                    else:
                        assert False
            else: # Snapshot graph. Build the entire graph from scratch.
                if args.verbose:
                    print("    building " + graph_name + " from scratch...")
                G = nx.MultiDiGraph()
                if not args.list:
                    for node in graph.find('nodes'):
                        parse_add_node(G, node, graph)
                    for edge in graph.find('edges'):
                        parse_add_edge(G, edge)
            if is_difference:
                previousG = G
            # TODO: Load the control-flow graph, if available.
            CFG = None
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
            et.SubElement(xml_node_properties, 'p', name=property).text = str(value)
    xml_edges = et.SubElement(xml_graph, 'edges')
    for e in list(G.edges):
        (src, dst, index) = e
        et.SubElement(xml_edges, 'edge', dict([('from', str(src)), ('to', str(dst)), ('index', str(index))]))
    et.SubElement(xml_graph, 'edges')

def create_sub_graph(args, G, original_G, with_neighbors_shown=True):
    if not args.nodes:
        return G
    nr_nodes = int(args.nodes)
    nodes = list(G.nodes)
    if not nodes or nr_nodes > len(nodes):
        return G
    while len(nodes) > nr_nodes:
        n = random.choice(nodes)
        G.remove_node(n)
        nodes.remove(n)
    if with_neighbors_shown:
        add_hidden_neighbors(G, original_G)
    return G

def add_hidden_node(G, original_G, node):
    if not original_G.has_node(node) or G.has_node(node):
        return
    G.add_node(node, **dict(original_G.nodes[node].items()), faded=True)
    # add edges
    edges = list(edge for edge in itertools.chain(original_G.in_edges(node, keys=True), original_G.out_edges(node, keys=True)))     # all edges to/from node
    edges = set([(a,b,c) for a,b,c in edges if G.has_node(a) and G.has_node(b) and not G.has_edge(a,b,c)])
    G.add_edges_from(edges)

def add_hidden_neighbors(G, original_G):
    nodes = list(G.nodes)
    for node in nodes:
        hidden_nodes = find_hidden_neighbors(G, original_G, node)
        for n in hidden_nodes:
            add_hidden_node(G, original_G, n)

def expand(args, graph, original_G, key, expanded_graphs):
    n = int(args.size)
    ((method, phase), G, CFG) = graph
    for k in range(key, key + n):
        G = G.copy()
        action = step(args, G, original_G)
        expanded_graphs[k] = ((method, action), G, CFG)
    return key + n

def step(args, G, original_G):
    # Simulate action on G (e.g. hiding or expansion).
    nodes = list(G.nodes)
    if not nodes:
        return
    if args.method == 'add':
        return add_one_node_with_edges(G, original_G)
    elif args.method == 'explore':
        func = random.choice([simulate_hiding_node, simulate_expanding_node])
        return func(G, original_G)
    else:
        print("unsupported expansion method", args.method)
        return

def add_one_node_with_edges(G, original_G):
    nodes = list(G.nodes)
    hidden = set([])
    for node in nodes:
        hidden.update(find_hidden_neighbors(G, original_G, node))
    node = random.choice(list(hidden))
    add_hidden_node(G, original_G, node)
    return "Added node " + str(node)

def find_hidden_neighbors(G, original_G, node):
    nodes = list(G.nodes)
    hidden = list(n for n in itertools.chain(original_G.predecessors(node), original_G.successors(node)) if not G.has_node(n))
    return hidden

def simulate_hiding_node(G, original_G):
    nodes = list(G.nodes)
    shown_nodes = list(n for n,faded in G.nodes(data="faded") if not faded)
    if not shown_nodes:
        return
    node = random.choice(shown_nodes)
    hide_shown_node(G, node)
    return "Hiding node " + str(node)

def hide_shown_node(G, node):
    if not G.has_node:
        return
    # remove all faded neighbors to <node>
    # make <node> faded
    nx.set_node_attributes(G, {node: {"faded": True}})
    faded_neighbors = list(n for n in itertools.chain(G.predecessors(node), G.successors(node)) if "faded" in G.nodes[n] and G.nodes[n]["faded"])
    shown_nodes = list(n for n,faded in G.nodes(data="faded") if not faded)
    for faded_node in faded_neighbors:
        if not any(faded_node in itertools.chain(G.successors(n), G.predecessors(n)) for n in shown_nodes) and G.has_node(faded_node):
            G.remove_node(faded_node)

def simulate_expanding_node(G, original_G):
    nodes = list(G.nodes)
    faded_nodes = list(n for n,faded in G.nodes(data="faded") if faded)
    if not nodes or not faded_nodes:
        return
    node = random.choice(faded_nodes)
    expand_faded_node(G, original_G, node)
    return "Expanding node " + str(node)

def expand_faded_node(G, original_G, node):
    nx.set_node_attributes(G, {node: {"faded": False}})
    neighbors = find_hidden_neighbors(G, original_G, node)
    for n in neighbors:
        add_hidden_node(G, original_G, n)


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
    list_filter.add_argument('--nodes',
                             metavar='N',
                             help="number of nodes the first graph should consist of")
    list_filter.add_argument('--sequences',
                             metavar='N',
                             help="number of sequences that should be expanded (default: all)")
    list_filter.add_argument('--method',
                             type=str,
                             default="explore",
                             help="graph expansion method, either 'add' or 'explore' (default: %(default)s)")

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
        seqs = 0
        max_seqs = int(args.sequences) if args.sequences else len(graphs.items())
        key = 0
        for (graph_id, ((method, phase), G, CFG)) in graphs.items():
            original_G = G.copy()
            method = method + "::" + str(graph_id)
            if args.verbose:
                print("expanding " + str(graph_id) + " " + method + "::" + phase + "...")
            initial_G = create_sub_graph(args, G, original_G)
            expanded_graphs[key] = ((method, phase), initial_G, CFG)
            key += 1
            key = expand(args, ((method, phase), initial_G, CFG), original_G, key, expanded_graphs)
            seqs += 1
            if seqs >= max_seqs:
                break
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
