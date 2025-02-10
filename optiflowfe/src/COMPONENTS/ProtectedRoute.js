import { Navigate, Outlet } from "react-router-dom";
import { useRecoilValue } from "recoil";

import { loginToken } from "../recoil/LoginAtom";

export default function ProtectedRoute() {
    const token = useRecoilValue(loginToken);
    return token ? <Outlet /> : <Navigate to ="/unauthorized" replace />;
}
