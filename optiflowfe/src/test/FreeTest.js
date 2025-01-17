
import { useCallback, useState } from 'react';
import {
    ReactFlow,
    addEdge,
    applyEdgeChanges,
    applyNodeChanges,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import CustomNode from "./CustomNode";
import DrainageNode from './DrainageNode';
import InterSectionNode from './InterSectionNode';
import NormalNode from "./NormalNode";

// const nodeTypes = { customN: CustomNode };

const nodeTypes = { drain: DrainageNode, intersection: InterSectionNode, normal: NormalNode };

const initialNodes = [
    {
        id: 'base',
        dragHandle : "false",
        position: { x: 0, y: 0 },
        data: { label: '정수지' },
    },

    {
        id: 'n1', type: 'intersection', position: { x: 0, y: 100 },
        data: { label: '1' },
    },

    {
        id: 'n2', type: 'intersection', position: { x: -100, y: 100 },
        data: { label: '2' },
    },

    {
        id: 'drainF', type: 'drain', position: { x: -200, y: 100 },
        data: { label: "F저수지", capacity:100, crtVol:10 },
    },

    {
        id: 'drainJ', type: 'drain', position: { x: 200, y: 100 },
        data: { label: "J저수지", value: 123, capacity:122, crtVol:55 },
    },

    {
        id: 'pAA', type: 'normal', position: { x: -100, y: 200 },
        data: { label: 'AA가압장' },
    },

    {
        id: 'n4', type: 'intersection', position: { x: -100, y: 300 },
        data: { label: '4' },
    },

    {
        id: 'n5', type: 'intersection', position: { x: -100, y: 400 },
        data: { label: '5' },
    },

    {
        id: 'n6', type: 'intersection', position: { x: -100, y: 500 },
        data: { label: '6' },
    },

    {
        id: 'n7', type: 'intersection', position: { x: -100, y: 600 },
        data: { label: '7' },
    },

    {
        id: 'n8', type: 'intersection', position: { x: -100, y: 700 },
        data: { label: '8' },
    },


    {
        id: 'n3', type: 'intersection', position: { x: 0, y: 200 },
        data: { label: '3' },
    },

];
const initialEdges = [
    { id: 'base-n1',    source: 'base',     target: 'n1', targetHandle:"t-target"  },
    { id: 'n1-n2',      source: 'n1',       target: 'n2', sourceHandle:"l-source", targetHandle:"r-target"},
    { id: 'n2-drainF',  source: 'n2',       target: 'drainF',  sourceHandle:"l-source", targetHandle:"r-target"},
    { id: 'n1-drainJ',  source: 'n1',       target: 'drainJ',  sourceHandle:"r-source", targetHandle:"l-target"},
    { id: 'n2-ppA',  source: 'n2',       target: 'pAA',  sourceHandle:"b-source", targetHandle:"t-target"},
    
];

export default function FreeTest() {
    const [nodes, setNodes] = useState(initialNodes);
    const [edges, setEdges] = useState(initialEdges);

    const onNodesChange = useCallback(
        (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
        [setNodes],
    );
    const onEdgesChange = useCallback(
        (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
        [setEdges],
    );
    const onConnect = useCallback(
        (connection) => setEdges((eds) => addEdge(connection, eds)),
        [setEdges],
    );

    return (
        <div className="w-full h-screen">

            <div className='size-full bg-blue-50'>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                    fitView
                />
            </div>
        </div>

    )
}
