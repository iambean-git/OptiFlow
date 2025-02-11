import React, { useState } from "react";

export default function AnimatedInput({ type, label, value, onChange, detail }) {
  const [isFocused, setIsFocused] = useState(false);
  const inputId = `input-${label.replace(/\s+/g, "-").toLowerCase()}`; // 고유한 ID 생성

  return (
    <div className={`relative w-80 h-16 ${isFocused || value ? "mt-3" :""}`}>
      {/* 라벨 (클릭 가능) */}
      <label
        htmlFor={inputId} // input과 연결
        className={`absolute cursor-text transition-all duration-300 z-10
          ${isFocused || value ? "left-1 top-0 text-xs text-gray-600" : "left-3 top-[30px] text-sm"} 
          ${isFocused ? "text-blue-500" : "text-gray-400"}
        `}
      >
        {label}
      </label>

      {/* 입력 필드 */}
      <input
        id={inputId} // label의 htmlFor와 연결
        type={type}
        value={value}
        onChange={onChange}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        className="absolute top-5 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-[#333]"
      />


      {/* 추가 설명 */}
      {detail && <div
        className={`absolute top-[30px] text-xs left-3 text-gray-400 mt-0.5
                  ${isFocused && !value ?  "block" : "hidden"}
        `}>{detail}</div>}
    </div>
  );
}
