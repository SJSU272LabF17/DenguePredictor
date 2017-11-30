<%-- <%@ page import="data.GetMapData" %><%
String strYear = request.getParameter("year");
String strWeek = request.getParameter("week");
int week = 0;

if(strWeek!= null && strWeek.trim().length()>0){
	week = Integer.parseInt(strWeek);
}
GetMapData data = new GetMapData();
String strData =  data.getDataArray(strYear, week);
%><%=strData%> --%>