import { memo } from "react";
import { BaseEdge, getSmoothStepPath } from '@xyflow/react';

const FlowEdge = memo(({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition }) => {
    const [edgePath] = getSmoothStepPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
    });

    return (
        <>
            <BaseEdge id={id} path={edgePath} />
            <circle r="3" fill="#3a6bd6 ">
                <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
            </circle>
        </>
    );
});

export default FlowEdge;
