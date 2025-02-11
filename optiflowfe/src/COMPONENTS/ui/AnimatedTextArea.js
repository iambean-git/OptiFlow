import React, { useState } from "react";

export default function AnimatedTextarea({ label, value, onChange, detail, rows = 3 }) {
  const [isFocused, setIsFocused] = useState(false);
  const inputId = `textarea-${label.replace(/\s+/g, "-").toLowerCase()}`;

  return (
    <div className={`relative w-80 ${isFocused || value ? "mt-3" : ""}`}>
      {/* 라벨 */}
      <label
        htmlFor={inputId}
        className={`absolute cursor-text transition-all duration-300 z-10
          ${isFocused || value ? "left-1 top-0 text-xs text-gray-600" : "left-3 top-[30px] text-sm"} 
          ${isFocused ? "text-blue-500" : "text-gray-400"}
        `}
      >
        {label}
      </label>

      {/* 텍스트 영역 */}
      <textarea
        id={inputId}
        value={value}
        onChange={onChange}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        rows={rows}
        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none mt-6"
      />

      {/* 추가 설명 */}
      {detail && (
        <div
          className={`absolute top-[30px] text-xs left-3 text-gray-400 mt-0.5
                  ${isFocused && !value ? "block" : "hidden"}
        `}
        >
          {detail}
        </div>
      )}
    </div>
  );
}
