import { Tooltip } from "react-tooltip";

export default function FreeTest() {
    return (
        <div className="flex flex-col items-center justify-center h-screen">
            <button
                data-tooltip-id="my-tooltip"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
                Hover me
            </button>

            <Tooltip
                id="my-tooltip"
                className="shadow-lg"
                style={{
                    backgroundColor: "#facc15", // Tailwind bg-yellow-200
                    color: "#374151", // Tailwind text-gray-300 (대신 #374151로 더 가시성 높임)
                    fontSize: "14px",
                    padding: "8px 12px",
                    borderRadius: "8px",
                }}
            >
                <div className="flex flex-col">
                    <div>J 정수장 </div>
                    <div>수위 28/70</div>
                    <div>저수량 2812</div>
                    <div>수용 가능 용량 2812</div>
                </div>
            </Tooltip>
        </div>
    );
}
