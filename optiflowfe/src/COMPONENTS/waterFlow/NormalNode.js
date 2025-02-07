import { Handle, Position } from '@xyflow/react';

export default function NormalNode({ data }) {
    return (
        <>
            <div className='flex bg-[#e6f4f1] rounded-sm justify-center items-center 
                            text-sm font-semibold px-4 py-2 text-[#333333] !z-0 relative'>
                {data.label}
            </div>

            <Handle
                type="source"
                position={Position.Top}
                id="t-source"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />


            <Handle
                type="target"
                position={Position.Top}
                id="t-target"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />


            <Handle
                type="source"
                position={Position.Bottom}
                id="b-source"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />
            <Handle
                type="target"
                position={Position.Bottom}
                id="b-target"
                className="!min-w-0 !min-h-0  !w-[2px] !h-[2px]  !border-0 !bg-[#00000000]"
            />
        </>
    )
}
