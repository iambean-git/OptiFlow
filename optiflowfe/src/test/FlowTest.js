import { ReactFlow, Background, Controls, applyEdgeChanges, applyNodeChanges, addEdge, useNodesState, useEdgesState } from '@xyflow/react';
import { useState, useCallback } from 'react';
import '@xyflow/react/dist/style.css';
import TextUpdateNode from './TextUpdateNode';
import CustomEdge from './CustomEdge';

const rfStyle = {
    backgroundColor: '#B8CEFF',
};

const initialNodes = [
    {
        id: 'node-1',
        type: 'textUpdater',
        position: { x: 0, y: 0 },
        data: { value: 123 },
    },
];

const nodeTypes = { textUpdater: TextUpdateNode };

const initialNodes2 = [
    { id: 'a', position: { x: 0, y: 0 }, data: { label: 'Node A' } },
    { id: 'b', position: { x: 0, y: 100 }, data: { label: 'Node B' } },
];

const initialEdges = [
    { id: 'a->b', type: 'custom-edge', source: 'a', target: 'b' },
];

const edgeTypes = {
    'custom-edge': CustomEdge,
};

export default function FlowTest() {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes2);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const onConnect = useCallback(
        (connection) => {
            const edge = { ...connection, type: 'custom-edge' };
            setEdges((eds) => addEdge(edge, eds));
        },
        [setEdges],
    );

    return (
        <div className='w-full h-screen bg-red-50'>
            <div className='w-full h-1/2'>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    edgeTypes={edgeTypes}
                    fitView
                />

            </div>

        </div>

    );

}
