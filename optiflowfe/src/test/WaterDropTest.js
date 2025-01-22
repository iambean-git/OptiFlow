import React from "react";

const WaterDrop = ({ path, dur }) => {
  return (
    <svg
      width="21"
      height="28"
      viewBox="0 0 45 60"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M22.2303 59.2349C9.97268 59.2349 0 49.8496 0 38.3136C0 22.2131 22.2304 0 22.2304 0C22.2304 0 44.4602 22.2131 44.4602 38.3136C44.4602 49.8497 34.4879 59.2349 22.2303 59.2349Z"
        fill="#3A6BD6"
      />
      {path && (
        <animateMotion dur={dur} repeatCount="indefinite" path={path} />
      )}
    </svg>
  );
};

export default WaterDrop;
