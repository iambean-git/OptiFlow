import { RiMapPinFill } from "react-icons/ri";
import NavBar from "../components/NavBar";
import React, { useEffect, useState, useRef } from "react";


export default function Regions() {

  return (
    <div className="w-full h-screen bg-[#f2f2f2] ">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col">
        <div className="w-full h-[160px] px-10 flex justify-between">
          {/* 텍스트 */}
          <div className="w-2/5 h-full  flex flex-col justify-end text-[#333333]">
            <h1 className="text-4xl ">타이틀(지역별)</h1>
            <p className="mt-2">각 배수지를 클릭하면, 세부 정보를 확인할 수 있습니다.</p>
          </div>

          {/* 달력 */}
          <div className="h-full bg-blue-50 relative  ">

          </div>

        </div>
        <section className="px-10 pb-10 pt-6 w-full h-full">
          <div className="w-full h-full border rounded-lg bg-white flex justify-center items-center relative">
            <img src="/images/map.png" className="w-10/12"></img>
            <RiMapPinFill className="text-indigo-500 text-2xl absolute left-[500px] top-[380px]"/>
          </div>
        </section>
      </div>
    </div>
  );
}
