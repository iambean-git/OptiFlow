import { useEffect, useState} from 'react';
import {
    ReactFlow,
    ReactFlowProvider,
    useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import "./waterFlow.css";

import DrainageNode from './DrainageNode';
import InterSectionNode from './InterSectionNode';
import NormalNode from './NormalNode';
import FlowEdge from './FlowEdge';
import Modal from '../modal/Modal';

const nodeTypes = { drain: DrainageNode, intersection: InterSectionNode, normal: NormalNode };
const edgeTypes = { flowEdge: FlowEdge };

const initialEdges = [
    { id: 'base-n1', source: 'base', target: 'n1', sourceHandle: "b-source", targetHandle: "t-target", animated: true },
    { id: 'n1-n2', source: 'n1', target: 'n2', sourceHandle: "l-source", targetHandle: "r-target", animated: true, },
    { id: 'n2-drainF', source: 'n2', target: 'drainF', sourceHandle: "l-source", targetHandle: "r-target", animated: true, type: "flowEdge" },
    { id: 'n1-drainJ', source: 'n1', target: 'drainJ', sourceHandle: "r-source", targetHandle: "l-target", animated: true, type: "flowEdge" },
    { id: 'n2-pAA', source: 'n2', target: 'pAA', sourceHandle: "b-source", targetHandle: "t-target", animated: true, },
    { id: 'pAA-n4', source: 'pAA', target: 'n4', sourceHandle: "b-source", targetHandle: "t-target", animated: true, },
    { id: 'n4-n5', source: 'n4', target: 'n5', sourceHandle: "l-source", targetHandle: "r-target", animated: true, },
    { id: 'n5-n6', source: 'n5', target: 'n6', sourceHandle: "l-source", targetHandle: "r-target", animated: true, },
    { id: 'n6-n7', source: 'n6', target: 'n7', sourceHandle: "l-source", targetHandle: "r-target", animated: true, },
    { id: 'n7-n8', source: 'n7', target: 'n8', sourceHandle: "l-source", targetHandle: "r-target", animated: true, },
    { id: 'n4-drainE', source: 'n4', target: 'drainE', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n5-drainD', source: 'n5', target: 'drainD', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n6-drainA', source: 'n6', target: 'drainA', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n7-drainB', source: 'n7', target: 'drainB', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n8-drainC', source: 'n8', target: 'drainC', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n1-n3', source: 'n1', target: 'n3', sourceHandle: "b-source", targetHandle: "t-target", animated: true, },
    { id: 'n3-n9', source: 'n3', target: 'n9', sourceHandle: "b-source", targetHandle: "t-target", animated: true, },
    { id: 'n9-drainG', source: 'n9', target: 'drainG', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n9-n10', source: 'n9', target: 'n10', sourceHandle: "r-source", targetHandle: "l-target", animated: true, },
    { id: 'n10-n11', source: 'n10', target: 'n11', sourceHandle: "b-source", targetHandle: "t-target", animated: true, },
    { id: 'n11-drainI', source: 'n11', target: 'drainI', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n11-n12', source: 'n11', target: 'n12', sourceHandle: "r-source", targetHandle: "l-target", animated: true, },
    { id: 'n12-drainH', source: 'n12', target: 'drainH', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
    { id: 'n12-n13', source: 'n12', target: 'n13', sourceHandle: "r-source", targetHandle: "l-target", animated: true, },
    { id: 'n13-pAB', source: 'n13', target: 'pAB', sourceHandle: "b-source", targetHandle: "t-target", animated: true, },
    { id: 'pAB-drainK', source: 'pAB', target: 'drainK', sourceHandle: "b-source", targetHandle: "l-target", animated: true, type: "flowEdge" },
    { id: 'n13-n14', source: 'n13', target: 'n14', sourceHandle: "r-source", targetHandle: "l-target", animated: true, },
    { id: 'n14-drainL', source: 'n14', target: 'drainL', sourceHandle: "b-source", targetHandle: "t-target", animated: true, type: "flowEdge" },
];

// ✅ `LayoutFlow`를 유지하면서 `fitView`를 적용하는 방식
const LayoutFlow = ({ nodes, edges, onNodeClick }) => {
    const { fitView } = useReactFlow();

    useEffect(() => {
        const handleResize = () => {
            fitView({ padding: 0.1 }); // fitView를 호출 (padding은 선택 사항)
        };

        // 초기 실행
        handleResize();

        // 윈도우 리사이즈 이벤트에 핸들러 등록
        window.addEventListener("resize", handleResize);

        return () => {
            // 컴포넌트 언마운트 시 이벤트 리스너 제거
            window.removeEventListener("resize", handleResize);
        };
    }, [fitView]);

    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            preventScrolling={false}
            panOnDrag={false}
        />
    );
};

function WaterFlowComponent({ waterLevel }) {
    const [nodes, setNodes] = useState([]);
    const [edges, setEdges] = useState(initialEdges);
    const [modalOpen, setModalOpen] = useState(false);
    const [modalData, setModalData] = useState('');

    const closeModal = () => setModalOpen(false);
    const onNodeClick = (event, node) => {
        if (node.type !== "drain") return;
        setModalData(node);
        setModalOpen(true);
    };

    useEffect(() => {
        // console.log("[WaterFlow] waterLevel 변경 감지", waterLevel);
        if (!waterLevel) return;

        const initialNodes = [
            {
                id: 'base', type: 'normal', position: { x: 0, y: 0 },
                dragHandle: "false",
                data: { label: '정수지' },
            },

            {
                id: 'n1', type: 'intersection', position: { x: 25, y: 80 },
                // dragHandle: "false",
                data: { label: '1' },
            },

            {
                id: 'n2', type: 'intersection', position: { x: -110, y: 80 },
                data: { label: '2' },
            },

            {
                id: 'drainF', type: 'drain', position: { x: -300, y: 34 },
                data: { label: "F배수지", capacity: waterLevel[1].reservoirId.capacity, maxHeight: waterLevel[1].reservoirId.height, crtHeight: waterLevel[1].height },
            },

            {
                id: 'drainJ', type: 'drain', position: { x: 214, y: 34 },
                data: { label: "J배수지", capacity: waterLevel[0].reservoirId.capacity, maxHeight: waterLevel[0].reservoirId.height, crtHeight: waterLevel[0].height },
            },

            {
                id: 'pAA', type: 'normal', position: { x: -147, y: 170 },
                data: { label: 'AA가압장' },
            },

            {
                id: 'n4', type: 'intersection', position: { x: -110, y: 270 },
                data: { label: '4' },
            },

            {
                id: 'drainE', type: 'drain', position: { x: -140, y: 370 },
                data: { label: "E배수지", capacity: waterLevel[1].reservoirId.capacity, maxHeight: waterLevel[2].reservoirId.height, crtHeight: waterLevel[2].height},
            },

            {
                id: 'n5', type: 'intersection', position: { x: -210, y: 270 },
                data: { label: '5' },
            },

            {
                id: 'drainD', type: 'drain', position: { x: -240, y: 370 },
                data: { label: "D배수지", capacity: waterLevel[3].reservoirId.capacity, maxHeight: waterLevel[3].reservoirId.height, crtHeight: waterLevel[3].height, },
            },

            {
                id: 'n6', type: 'intersection', position: { x: -310, y: 270 },
                data: { label: '6' },
            },

            {
                id: 'drainA', type: 'drain', position: { x: -340, y: 370 },
                data: { label: "A배수지", capacity: waterLevel[4].reservoirId.capacity, maxHeight: waterLevel[4].reservoirId.height, crtHeight: waterLevel[4].height, },
            },

            {
                id: 'n7', type: 'intersection', position: { x: -410, y: 270 },
                data: { label: '7' },
            },

            {
                id: 'drainB', type: 'drain', position: { x: -440, y: 370 },
                data: { label: "B배수지", capacity: waterLevel[5].reservoirId.capacity, maxHeight: waterLevel[5].reservoirId.height, crtHeight: waterLevel[5].height, },
            },

            {
                id: 'n8', type: 'intersection', position: { x: -510, y: 270 },
                data: { label: '8' },
            },

            {
                id: 'drainC', type: 'drain', position: { x: -540, y: 370 },
                data: { label: "C배수지", capacity: waterLevel[6].reservoirId.capacity, maxHeight: waterLevel[6].reservoirId.height, crtHeight: waterLevel[6].height, },
            },


            {
                id: 'n3', type: 'intersection', position: { x: 25, y: 178 },
                data: { label: '3' },
            },

            {
                id: 'n9', type: 'intersection', position: { x: 25, y: 270 },
                data: { label: '9' },
            },

            {
                id: 'drainG', type: 'drain', position: { x: 40, y: 410 },
                data: { label: "G배수지", capacity: waterLevel[7].reservoirId.capacity, maxHeight: waterLevel[7].reservoirId.height, crtHeight: waterLevel[7].height, },
            },

            {
                id: 'n10', type: 'intersection', position: { x: 190, y: 270 },
                data: { label: '10' },
            },

            {
                id: 'n11', type: 'intersection', position: { x: 190, y: 370 },
                data: { label: '11' },
            },

            {
                id: 'drainI', type: 'drain', position: { x: 160, y: 480 },
                data: { label: "I배수지", capacity: waterLevel[8].reservoirId.capacity, maxHeight: waterLevel[8].reservoirId.height, crtHeight: waterLevel[8].height, },
            },

            {
                id: 'n12', type: 'intersection', position: { x: 290, y: 370 },
                data: { label: '12' },
            },

            {
                id: 'drainH', type: 'drain', position: { x: 260, y: 480 },
                data: { label: "H배수지", capacity: waterLevel[9].reservoirId.capacity, maxHeight: waterLevel[9].reservoirId.height, crtHeight: waterLevel[9].height, },
            },

            {
                id: 'n13', type: 'intersection', position: { x: 390, y: 370 },
                data: { label: '13' },
            },

            {
                id: 'pAB', type: 'normal', position: { x: 354, y: 440 },
                data: { label: 'AB가압장' },
            },

            {
                id: 'drainK', type: 'drain', position: { x: 495, y: 480 },
                data: { label: "K배수지", capacity: waterLevel[11].reservoirId.capacity, maxHeight: waterLevel[11].reservoirId.height, crtHeight: waterLevel[11].height, },
            },

            {
                id: 'n14', type: 'intersection', position: { x: 625, y: 370 },
                data: { label: '14' },
            },

            {
                id: 'drainL', type: 'drain', position: { x: 595, y: 480 },
                data: { label: "L배수지", capacity: waterLevel[10].reservoirId.capacity, maxHeight: waterLevel[10].reservoirId.height, crtHeight: waterLevel[10].height, },
            },
        ];

        setNodes((prevNodes) => JSON.stringify(prevNodes) === JSON.stringify(initialNodes) ? prevNodes : initialNodes);
    }, [waterLevel]);

    return (
        <div className="w-full h-full rounded-lg bg-white" style={{ boxShadow: "0px 0px 15px rgba(0, 0, 0, 0.15)" }}>
            <LayoutFlow nodes={nodes} edges={edges} onNodeClick={onNodeClick} />
            <Modal open={modalOpen} close={closeModal} data={modalData} />
        </div>
    );
}

// ✅ `ReactFlowProvider`를 최상위에서 감싸고, 내부 컴포넌트는 변화가 있을 때만 리렌더링!
export default function WaterFlow(props) {
    return (
        <ReactFlowProvider>
            <WaterFlowComponent {...props} />
        </ReactFlowProvider>
    );
}
