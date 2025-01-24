import { useCallback, useEffect, useState } from "react";

import {
  ReactFlow,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  useEdges,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import DrainageNode from "./DrainageNode";
import InterSectionNode from "./InterSectionNode";
import NormalNode from "./NormalNode";

import FlowEdge from "./FlowEdge";

import Modal from "../modal/Modal";

const nodeTypes = {
  drain: DrainageNode,
  intersection: InterSectionNode,
  normal: NormalNode,
};
const edgeTypes = { flowEdge: FlowEdge };

const initialEdges = [
  {
    id: "base-n1",
    source: "base",
    target: "n1",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
  },
  {
    id: "n1-n2",
    source: "n1",
    target: "n2",
    sourceHandle: "l-source",
    targetHandle: "r-target",
    animated: true,
  },
  {
    id: "n2-drainF",
    source: "n2",
    target: "drainF",
    sourceHandle: "l-source",
    targetHandle: "r-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n1-drainJ",
    source: "n1",
    target: "drainJ",
    sourceHandle: "r-source",
    targetHandle: "l-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n2-pAA",
    source: "n2",
    target: "pAA",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
  },
  {
    id: "pAA-n4",
    source: "pAA",
    target: "n4",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
  },
  {
    id: "n4-n5",
    source: "n4",
    target: "n5",
    sourceHandle: "l-source",
    targetHandle: "r-target",
    animated: true,
  },
  {
    id: "n5-n6",
    source: "n5",
    target: "n6",
    sourceHandle: "l-source",
    targetHandle: "r-target",
    animated: true,
  },
  {
    id: "n6-n7",
    source: "n6",
    target: "n7",
    sourceHandle: "l-source",
    targetHandle: "r-target",
    animated: true,
  },
  {
    id: "n7-n8",
    source: "n7",
    target: "n8",
    sourceHandle: "l-source",
    targetHandle: "r-target",
    animated: true,
  },
  {
    id: "n4-drainE",
    source: "n4",
    target: "drainE",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n5-drainD",
    source: "n5",
    target: "drainD",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n6-drainA",
    source: "n6",
    target: "drainA",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n7-drainB",
    source: "n7",
    target: "drainB",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n8-drainC",
    source: "n8",
    target: "drainC",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n1-n3",
    source: "n1",
    target: "n3",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
  },
  {
    id: "n3-n9",
    source: "n3",
    target: "n9",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
  },
  {
    id: "n9-drainG",
    source: "n9",
    target: "drainG",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n9-n10",
    source: "n9",
    target: "n10",
    sourceHandle: "r-source",
    targetHandle: "l-target",
    animated: true,
  },
  {
    id: "n10-n11",
    source: "n10",
    target: "n11",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
  },
  {
    id: "n11-drainI",
    source: "n11",
    target: "drainI",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n11-n12",
    source: "n11",
    target: "n12",
    sourceHandle: "r-source",
    targetHandle: "l-target",
    animated: true,
  },
  {
    id: "n12-drainH",
    source: "n12",
    target: "drainH",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n12-n13",
    source: "n12",
    target: "n13",
    sourceHandle: "r-source",
    targetHandle: "l-target",
    animated: true,
  },
  {
    id: "n13-pAB",
    source: "n13",
    target: "pAB",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
  },
  {
    id: "pAB-drainK",
    source: "pAB",
    target: "drainK",
    sourceHandle: "b-source",
    targetHandle: "l-target",
    animated: true,
    type: "flowEdge",
  },
  {
    id: "n13-n14",
    source: "n13",
    target: "n14",
    sourceHandle: "r-source",
    targetHandle: "l-target",
    animated: true,
  },
  {
    id: "n14-drainL",
    source: "n14",
    target: "drainL",
    sourceHandle: "b-source",
    targetHandle: "t-target",
    animated: true,
    type: "flowEdge",
  },
];

export default function WaterFlow() {
  // ================================= 모달 관련 start =================================
  const [modalOpen, setModalOpen] = useState(false);
  const [modalData, setModalData] = useState("");

  const closeModal = () => {
    setModalOpen(false);
  };

  //각 노드를 클릭했을 때
  const onNodeClick = (event, node) => {
    console.log("Clicked node:", node);

    //노드 유형이 drain일 때만 모달팝업 오픈
    if (node.type !== "drain") return;
    setModalData(node);
    setModalOpen(true);
  };
  // ================================= 모달 관련 end =================================

  const [waterLevel, setWaterLevel] = useState("");
  const [reservoirInfo, setReservoirInfo] = useState();

  const [nodes, setNodes] = useState("");
  const [edges, setEdges] = useState(initialEdges);

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes]
  );
  const onEdgesChange = useCallback(
    (changes) => {
      setEdges((eds) => applyEdgeChanges(changes, eds));
    },

    [setEdges]
  );
  const onConnect = useCallback(
    (connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );

  const handleNodeDragStop = (event, node) => {
    console.log(`Node ${node.id} final position:`, node.position);
  };

  useEffect(() => {
    console.log("패치합니당:");
    fetchWaterLevel();
    fetchReservoirInfo();
    // fetchdata33();
  }, []);

  const fetchWaterLevel = async () => {
    const url =
      "http://10.125.121.226:8080/api/waterlevels/2023-10-21T00:01:00";

    const resp = await fetch(url);
    const data = await resp.json(); // JSON 형식으로 응답 받기
    console.log("수위 data :", data);
    setWaterLevel(data);
  };

  const fetchReservoirInfo = async () => {
    const url = "http://10.125.121.226:8080/api/reservoirs";
    const resp = await fetch(url);
    const data = await resp.json(); // JSON 형식으로 응답 받기
    // console.log("fetchReservoirInfo :", data);
    setReservoirInfo(data);

    // const {resp} = await axios.get(url);
    // console.log("resp:", resp);

    // fetch("http://10.125.121.226:8080/api/reservoirs")
    //     .catch(error => console.error('Error:', error));
  };

  useEffect(() => {
    if (!waterLevel) return;
    if (!reservoirInfo) return;

    console.log("waterLevel : ", waterLevel);
    console.log("reservoirInfo : ", reservoirInfo);

    const initialNodes = [
      {
        id: "base",
        type: "normal",
        position: { x: 0, y: 0 },
        dragHandle: "false",
        onclick: console.log("click"),
        data: { label: "정수지" },
      },

      {
        id: "n1",
        type: "intersection",
        position: { x: 25, y: 80 },
        // dragHandle: "false",
        data: { label: "1" },
      },

      {
        id: "n2",
        type: "intersection",
        position: { x: -110, y: 80 },
        data: { label: "2" },
      },

      {
        id: "drainF",
        type: "drain",
        position: { x: -300, y: 34 },
        data: {
          label: "F배수지",
          capacity: reservoirInfo[1].capacity,
          maxHeight: reservoirInfo[1].height,
          crtHeight: waterLevel[0]["j"],
          crtVol: 10,
        },
      },

      {
        id: "drainJ",
        type: "drain",
        position: { x: 214, y: 34 },
        data: {
          label: "J배수지",
          capacity: reservoirInfo[0].capacity,
          maxHeight: reservoirInfo[0].height,
          crtHeight: waterLevel[0]["j"],
          crtVol: 55,
        },
      },

      {
        id: "pAA",
        type: "normal",
        position: { x: -147, y: 170 },
        data: { label: "AA가압장" },
      },

      {
        id: "n4",
        type: "intersection",
        position: { x: -110, y: 270 },
        data: { label: "4" },
      },

      {
        id: "drainE",
        type: "drain",
        position: { x: -140, y: 370 },
        data: {
          label: "E배수지",
          capacity: reservoirInfo[2].capacity,
          maxHeight: reservoirInfo[2].height,
          crtHeight: waterLevel[0]["e"],
          crtVol: 35,
        },
      },

      {
        id: "n5",
        type: "intersection",
        position: { x: -210, y: 270 },
        data: { label: "5" },
      },

      {
        id: "drainD",
        type: "drain",
        position: { x: -240, y: 370 },
        data: {
          label: "D배수지",
          capacity: reservoirInfo[3].capacity,
          maxHeight: reservoirInfo[3].height,
          crtHeight: waterLevel[0]["d"],
          crtVol: 35,
        },
      },

      {
        id: "n6",
        type: "intersection",
        position: { x: -310, y: 270 },
        data: { label: "6" },
      },

      {
        id: "drainA",
        type: "drain",
        position: { x: -340, y: 370 },
        data: {
          label: "A배수지",
          capacity: reservoirInfo[4].capacity,
          maxHeight: reservoirInfo[4].height,
          crtHeight: waterLevel[0]["a"],
          crtVol: 120,
        },
      },

      {
        id: "n7",
        type: "intersection",
        position: { x: -410, y: 270 },
        data: { label: "7" },
      },

      {
        id: "drainB",
        type: "drain",
        position: { x: -440, y: 370 },
        data: {
          label: "B배수지",
          capacity: reservoirInfo[5].capacity,
          maxHeight: reservoirInfo[5].height,
          crtHeight: waterLevel[0]["b"],
          crtVol: 120,
        },
      },

      {
        id: "n8",
        type: "intersection",
        position: { x: -510, y: 270 },
        data: { label: "8" },
      },

      {
        id: "drainC",
        type: "drain",
        position: { x: -540, y: 370 },
        data: {
          label: "C배수지",
          capacity: reservoirInfo[6].capacity,
          maxHeight: reservoirInfo[6].height,
          crtHeight: waterLevel[0]["c"],
          crtVol: 120,
        },
      },

      {
        id: "n3",
        type: "intersection",
        position: { x: 25, y: 178 },
        data: { label: "3" },
      },

      {
        id: "n9",
        type: "intersection",
        position: { x: 25, y: 270 },
        data: { label: "9" },
      },

      {
        id: "drainG",
        type: "drain",
        position: { x: 40, y: 410 },
        data: {
          label: "G배수지",
          capacity: reservoirInfo[7].capacity,
          maxHeight: reservoirInfo[7].height,
          crtHeight: waterLevel[0]["g"],
          crtVol: 20,
        },
      },

      {
        id: "n10",
        type: "intersection",
        position: { x: 190, y: 270 },
        data: { label: "10" },
      },

      {
        id: "n11",
        type: "intersection",
        position: { x: 190, y: 370 },
        data: { label: "11" },
      },

      {
        id: "drainI",
        type: "drain",
        position: { x: 160, y: 480 },
        data: {
          label: "I배수지",
          capacity: reservoirInfo[8].capacity,
          maxHeight: reservoirInfo[8].height,
          crtHeight: waterLevel[0]["i"],
          crtVol: 20,
        },
      },

      {
        id: "n12",
        type: "intersection",
        position: { x: 290, y: 370 },
        data: { label: "12" },
      },

      {
        id: "drainH",
        type: "drain",
        position: { x: 260, y: 480 },
        data: {
          label: "H배수지",
          capacity: reservoirInfo[9].capacity,
          maxHeight: reservoirInfo[9].height,
          crtHeight: waterLevel[0]["h"],
          crtVol: 20,
        },
      },

      {
        id: "n13",
        type: "intersection",
        position: { x: 390, y: 370 },
        data: { label: "13" },
      },

      {
        id: "pAB",
        type: "normal",
        position: { x: 354, y: 440 },
        data: { label: "AB가압장" },
      },

      {
        id: "drainK",
        type: "drain",
        position: { x: 495, y: 480 },
        data: {
          label: "K배수지",
          capacity: reservoirInfo[11].capacity,
          maxHeight: reservoirInfo[11].height,
          crtHeight: waterLevel[0]["k"],
          crtVol: 70,
        },
      },

      {
        id: "n14",
        type: "intersection",
        position: { x: 625, y: 370 },
        data: { label: "14" },
      },

      {
        id: "drainL",
        type: "drain",
        position: { x: 595, y: 480 },
        data: {
          label: "L배수지",
          capacity: reservoirInfo[10].capacity,
          maxHeight: reservoirInfo[10].height,
          crtHeight: waterLevel[0]["l"],
          crtVol: 20,
        },
      },
    ];
    setNodes(initialNodes);
  }, [waterLevel, reservoirInfo]);

  return (
    <div className="w-full h-full ">
      <div className="size-full ">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onNodeClick={onNodeClick}
          onEdgesChange={onEdgesChange}
          onNodeDragStop={handleNodeDragStop}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
        />
      </div>

      <Modal open={modalOpen} close={closeModal} data={modalData} />
    </div>
  );
}
