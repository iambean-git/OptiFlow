import { useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';

export default function InterSectionNode({ data }) {
    return (
        <>
            <div className='w-7 h-7 flex justify-center items-center'>
                <div className='size-5 border-black border rounded-full text-center text-xs'>
                    {data.label}
                </div>

            </div>

            <Handle
                type="source"
                position={Position.Top}
                id="t-source"
                className="w-16 !bg-teal-500"
            />
            <Handle
                type="target"
                position={Position.Top}
                id="t-target"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="source"
                position={Position.Bottom}
                id="b-source"
                className="w-16 !bg-teal-500"
            />
            <Handle
                type="target"
                position={Position.Bottom}
                id="b-target"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="source"
                position={Position.Right}
                id="r-source"
                className="w-16 !bg-teal-500"
            />
            <Handle
                type="target"
                position={Position.Right}
                id="r-target"
                className="w-16 !bg-teal-500"
            />

            <Handle
                type="source"
                position={Position.Left}
                id="l-source"
                className="w-16 !bg-teal-500"
            />
            <Handle
                type="target"
                position={Position.Left}
                id="l-target"
                className="w-16 !bg-teal-500"
            />
        </>
    )
}
