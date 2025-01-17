import { Handle, Position } from '@xyflow/react';

export default function NormalNode({ data }) {
    return (
        <>
            <div className='w-12 h-5 flex bg-gray-300 rounded-sm justify-center items-center
                            text-[10px]'>
                {data.label}
            </div>

            <Handle
                type="source"
                position={Position.Top}
                id="t-source"
                className=""
            />

            <Handle
                type="target"
                position={Position.Top}
                id="t-target"
                className=""
            />


            <Handle
                type="source"
                position={Position.Bottom}
                id="b-source"
                className=""
            />
            <Handle
                type="target"
                position={Position.Bottom}
                id="b-target"
                className=""
            />
        </>
    )
}
