import numpy as np
import networkx as nx
from NPort import NPort
import matplotlib.pyplot as mpl

class Circuit(object):
    net_number = 0

    def __init__(self):
        super(Circuit, self).__init__()
        self.circuit_graph = nx.Graph()         # Create empty circuit
        self.sfg = nx.DiGraph()

    # Functions for creating the netlist
    def add_port(self, port):
        nports = port.nports
        port_names = ['net_#{:d}'.format(n) for n in range(nports)]
        source = nports*[True]
        self.circuit_graph.add_node(port, portmap=port_names, source=source)

    def connect_port(self, device, port_num, netname=None):
        """
        Connect the selected device of the specified NPort components to the given
        netname. When the given Nports are not present in the circuit they will be
        added.

        Currently the function only works when exactly two NPort components are
        given. In the future the function will be extended to include any number
        of device. This will be handled by dynamically adding T-sections between
        the different device.

        In case one port is given the portnumber is simply tied to the given netname
        and no connection is made between NPort components.

        Usage:
            connect_port(device, port_num, netname)
        Parameters:
            device:      list of Nport components to be connected
            port_num:   list of the port numbers to connect to the netname
            netname:    netname of the connection. When no netname is
                        given a numbered netname will be generated. Currently it
                        is not checked for uniqueness!

        :param device: list
        :param port_num: list
        :param netname: string
        """
        if (not isinstance(device, list)) or (not isinstance(port_num, list)):
            return

        if len(device) != len(port_num):
            print '!! Circuit.connect_port number of device and port numbers should be the same'
            return -1

        # For now only two device are possible. T-sections should be added manually
        if (len(device) == 0) or (len(device) > 2):
            print '!! Circuit.connect_port can only handle two device at the moment'
            return -1

        # No netname given: add general unique number (static var)
        if netname is None:
            netname = 'net_#{:d}'.format(self.net_number)
            self.net_number += 1

        source_port = True
        for (port_, num_) in zip(device, port_num):
            # Add device to the circuit if they are not part of it
            if not self.circuit_graph.has_node(port_):
                # Port is not in the current circuit -> add
                self.add_port(port_)

            # Port exists, connect to netname
            self.circuit_graph.node[port_]['portmap'][num_] = netname
            # Should still add the source, dest property of the nodes.
            # This is needed to properly connect the a and b waves, say
            # NPort1 is connected to NPort2 then b2 of NPort1 connects
            # to a1 of NPort1
            self.circuit_graph.node[port_]['source'][num_] = source_port
            source_port = not source_port   # Flip source /dest

        # Add edge between the nodes (only works when two nodes are given
        if len(device) == 2:
            self.circuit_graph.add_edge(device[0], device[1], netname=netname)

    # Signal Flow Graph functions
    def create_sgf(self, freq=None, threshold=0.0):
        # Empty the current sfg
        # Loop through list of nodes in netlist
        #   Create sfg of each node (NPort)
        #       Pass the nport a list of netnames
        #       The NPort will create the SFG with ports netname_a ; netname_b
        #   Add the NPort sfg to the circuit sfg
        self.sfg.clear()

        for n, node in enumerate(self.circuit_graph.nodes(data=True)):
            # Index of the node:
            #   0: node itself
            #   1: properties
            # Create sfg of the selected node
            portmap = node[1]['portmap']
            source = node[1]['source']
            #print portmap
            node[0].create_sfg(freq=freq, portmap=portmap, sourcemap=source, threshold=threshold)

            #print 'Neighbors: {}'.format(self.circuit_graph.neighbors(node[0]))

            # update the position of the nodes (within the Nport they are only local coordinates)
            print 'Nodes to be added:'
            print node[0].sfg.nodes(data=True)
            print '/Nodes to be added:'

            # Add nodes sfg to the global one
            self.sfg.add_nodes_from(node[0].sfg.nodes(data=True))
            self.sfg.add_edges_from(node[0].sfg.edges(data=True))

    def convert_portdesc_nodename(self, portdesc):
        portmap = self.circuit_graph.node[portdesc['device']]['portmap']
        sourcemap = self.circuit_graph.node[portdesc['device']]['source']
        netname = portmap[portdesc['port_num']]
        if (portdesc['direction'] == 'a') or (portdesc['direction'] == 'in'):
            if sourcemap[ portdesc['port_num'] ]:
                # the port is specified to be a source: do not invert a and b
                ext = '_a'
            else:
                ext = '_b'
        else:
            if sourcemap[ portdesc['port_num'] ]:
                # the port is specified to be a source: do not invert a and b
                ext = '_b'
            else:
                ext = '_a'
        nodename = netname + ext

        return nodename

    def calc_transfer(self, source, sink, freq=None):
        """
        Parameters:
            source: dictionary describing the source port
            sink:   dictionary describing the sink port

            source/sink:
                key             description
                ['device']      Device to look 'into' (NPort class)
                ['port_num']    Port number (start counting from 0)
                ['direction']   in or a: input power wave
                                out or b: output power wave
        :param source:
        :param sink:
        :return:
        """
        print '-> Circuit.calc_transfer'
        # parameters should be dictionaries
        if (not isinstance(source, dict)) or (not isinstance(sink, dict)):
            return -1

        # parameters should contain the keys: 'device', 'port_num', 'direction'
        for par in [source, sink]:
            if not par.has_key('device'):
                print '!! Circuit.calc_transfer: the source and sink dictionaries should have the "device" key'
                return -2
            if not par.has_key('port_num'):
                print '!! Circuit.calc_transfer: the source and sink dictionaries should have the "port_num" key'
                return -2
            if not par.has_key('direction'):
                print '!! Circuit.calc_transfer: the source and sink dictionaries should have the "direction" key'
                return -2

        # Get the corresponding nets / node names
        source_nodename = self.convert_portdesc_nodename(source)
        print '** Source_nodename: {}'.format(source_nodename)
        sink_nodename = self.convert_portdesc_nodename(sink)
        print '** Sink_nodename: {}'.format(sink_nodename)

        # Implementation of Mason's rule:
        # Create the signal flow graph
        self.create_sgf(freq)
        # First get all direct paths between source and sink
        direct_paths = list(nx.all_simple_paths(self.sfg, source_nodename, sink_nodename))
        print '** Direct paths found:'
        for path in direct_paths:
            print '\t{}'.format(path)

        # Get all first order loops
        loops = list()
        loops += [list(nx.simple_cycles(self.sfg))]
        print '** First order loops found:'
        for loop in loops[0]:
            print '\t{}'.format(loop)
        # Convert loops to list of sets and calculate the loopgain
        loop_sets = list()
        loopgain = list()
        for loop in loops[0]:
            # Store the set
            loop_sets += [set(loop)]
            # Calculate the loopgain
            loopgain_ = 1           # Start with a loopgain of 1
            node_prev = loop[0]     # Previous node
            for node in loop[1:]:
                # find edge between node_prev and node
                loopgain_ *= self.sfg.edge[node_prev][node]['transfer']
                node_prev = node
            # Store the loopgain
            loopgain += [loopgain_]

        # From the first order loops determine all higher order loops
        # Increase the order until there are no loops left (no max order at this time)
        # Could add a max order, or remove loops when the loopgain < threshold
        all_loops = self.loop_Norder(loop_1=loop_sets, loopgain_1=loopgain)
        print '\t-> All N-order loops found: {}'.format(all_loops)

        # Calculate the denomenator
        # Denom = 1 - sum(1st order loop gain) + sum(2nd order) - sum(3rd order) + etc

        # From the simple nth order loops create a list of loops per direct path

        print '-> END Circuit.calc_transfer'

    @staticmethod
    def loop_Norder(loop_1, loopgain_1=None, loopgain_Nm=None, loop_Nm=None):
        """
        Function to filter higher order loops given a set of 1st order loops and the N-1 order loops.

        An N-order loop is a combination of a 1st order loop and N-1 order loop where none of the nodes touch.

        Example:
            1st order loops:    ABC BCD DE FGH
            2nd order:          ABCDE ABCFGH BCDFGH
            3rd order:          ABCDEFGH
            4th order:          none

        The given parameters are supposed to be "lists of sets".

        :param loop_1: set
        :param loop_N: set
        :return:
        """
        print '\t-> Circuit.loop_Norder(loop_1={}, loop_Nm={})'.format(loop_1, loop_Nm)
        # Set the N-1 (Nminus) order set to the first order set, when none is given
        loop_Nm_none = False
        if loop_Nm == None:
            loop_Nm = loop_1
            loopgain_Nm = loopgain_1
            loop_Nm_none = True

        # Empty list of N-order sets
        loop_N = list()
        loopgain_N = list()
        # Loop through all N-1 loops and see if they are part of N-order loops
        for n, loopNm_ in enumerate(loop_Nm):
            # In case only first order loops (loop_1) are given don't check double occurences
            # Say: loop_1 = AB BC CD
            # Only check AB with BC and CD
            # Only check BC with CD
            # (Do not check CD with AB and BC, otherwise the double loop will occur twice)
            if loop_Nm_none:
                idx = n+1
            else:
                idx = 0
            for l, loop1_ in enumerate(loop_1[idx:]):
                # Calculate the intersection of the current first order loop. If
                # the intersection is empty we know that the loops don't touch
                # and we have found an N-order loop
                if loopNm_.isdisjoint(loop1_):
                    loop_N += [loopNm_.union(loop1_)]   # Join set of the first order loop loop1_ and the matched loopNm_
                    if (loopgain_1 != None) and (loopgain_Nm != None):
                        loopgain_N += [loopgain_1[l+idx]*loopgain_Nm[n]]
                        print '\t\t** N_order loopgain: {}'.format(loopgain_N[-1])
                    print '\t\t** Found an N_order loop: {}'.format(loop_N[-1])

        # Did we find any N-order loops? If so search for N+1 order loops
        if len(loop_N) > 0:
            return [Circuit.loop_Norder(loop_1=loop_1, loopgain_1=loopgain_1, loop_Nm=loop_N, loopgain_Nm=loopgain_N), loop_Nm]

        print '\t-> END Circuit.loop_Norder'
        # Only first order loops found
        return [loop_Nm]

    # Print the netlist
    def print_circuit(self):
        print '** Circuit.print_circuit:'
        print '-- Nodes'
        print self.circuit_graph.nodes(data = True)
        print '-- Edges'
        print self.circuit_graph.edges(data=True)
        print '** /Circuit.print_circuit'

# Helper functions

if __name__ == "__main__":
    # Test loops:
    loops = [set('ab'), set('bc'), set('cd'), set('de'), set('ef')]
    print 'All loops of test: {}'.format(Circuit.loop_Norder(loops))

    # Test circuit
    circ1 = Circuit()
    port1 = NPort()
    port2 = NPort()
    port3 = NPort()
    port1.S=np.matrix(np.array([ [1, 1], [1, 1] ]))
    port2.S=np.matrix(np.array([ [2, 2], [2, 2] ]))
    port3.S=np.matrix(np.array([ [1, 1], [1, 1] ]))

    #circ1.add_port(port1)
    #circ1.add_port(port2)

    circ1.connect_port(device=[port1, port2], port_num=[1, 0], netname='mid12')
    circ1.connect_port(device=[port2, port3], port_num=[1, 0], netname='mid23')
    circ1.connect_port(device=[port1], port_num=[0], netname='in')
    circ1.connect_port(device=[port3], port_num=[1], netname='out')

    #circ1.print_circuit()
    # Calculate the transfer function between the input and output
    source = {}
    source['device'] = port1
    source['port_num'] = 0
    source['direction'] = 'in'
    sink = {}
    sink['device'] = port2
    sink['port_num'] = 1
    sink['direction'] = 'out'
    print ''
    circ1.calc_transfer(source, sink)

    #
    # print '-- Nodes'
    # print circ1.circuit_graph.nodes(data = True)
    # print '-- Edges'
    # print circ1.circuit_graph.edges(data=True)

    print nx.spring_layout(circ1.sfg)
    #nx.draw_spring(circ1.sfg, with_labels=True)
    #mpl.show()

    # SFG test
    #circ1.create_sgf()
    #print circ1.sfg.nodes(data=True)
    #print circ1.sfg.edges(data=True)
    #nx.draw(circ1.sfg, with_labels=True)
    #mpl.show()

    # # Port1
    # port1.S=np.matrix(np.array([ [1, 0], [0, 1] ]))
    # port1.create_sfg(portmap=['net1', 'net2'])
    # print port1.sfg.edges(data=True)
    # nx.draw(port1.sfg, with_labels=True)
    # mpl.show()
