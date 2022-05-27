def generate_bart_config(header, error_cols, error_fractions, error_strategy):
	for e in range(len(error_cols)):
		path = '/tmp/dirty_person.csv'
		if e == 0:
			path = '/tmp/generated_data.csv'

		start = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + \
				"<task>\n" + \
				"    <!-- ****************************************************\n" + \
				"                      DATABASES\n" + \
				"    **************************************************** -->\n" + \
				"    <source>\n" + \
				"        <type>DBMS</type>\n" + \
				"        <access-configuration>\n" + \
				"            <driver>org.postgresql.Driver</driver>\n" + \
				"            <uri>jdbc:postgresql:bart_example</uri>\n" + \
				"            <schema>source</schema>\n" + \
				"            <login>felix</login>\n" + \
				"            <password>felix</password>\n" + \
				"        </access-configuration>\n" + \
				"        <import>\n" + \
				"            <input type=\"csv\" separator=\",\" table=\"md\">"+ path +"</input>\n" + \
				"        </import>\n" + \
				"    </source>\n" + \
				"\t\n" + \
				"    <target> \n" + \
				"        <type>DBMS</type>\n" + \
				"        <access-configuration>\n" + \
				"            <driver>org.postgresql.Driver</driver>\n" + \
				"            <uri>jdbc:postgresql:bart_example</uri>\n" + \
				"            <schema>target</schema>\n" + \
				"            <login>felix</login>\n" + \
				"            <password>felix</password>\n" + \
				"        </access-configuration>\n" + \
				"        <import>\n" + \
				"            <input type=\"csv\" separator=\",\" table=\"person\">"+ path +"</input>\n" + \
				"        </import>\n" + \
				"    </target>\n" + \
				"\n" + \
				"    <!-- ****************************************************\n" + \
				"                    DEPENDENCIES\n" + \
				"    **************************************************** -->\n" + \
				"    <dependencies>\n" + \
				"<![CDATA[\n" + \
				"DCs:\n"


		fd = "person("
		for i in range(len(header)):
			fd += header[i] + ": $" + header[i] + "1, "
		fd = fd[:-2]
		fd += "),\nperson("
		for i in range(len(header)):
			fd += header[i] + ": $" + header[i] + "2, "
		fd = fd[:-2]
		fd += "),\n$" + header[0] + "1 == $" + header[0] +"2, $" + header[1] +"1 != $" + header[1] +"2 -> #fail.\n" + \
				"]]>\n"

		conf =  "    </dependencies>\n" + \
				"\n" + \
				"    <!-- ****************************************************\n" + \
				"                      CONFIGURATION\n" + \
				"    **************************************************** -->\n" + \
				"    <configuration>\n" + \
			    "        <!-- To print extra information in the ouput (default = false) -->\n" + \
				"        <printLog>true</printLog>\n" + \
				"\n" + \
				"        <!-- To load DB every time on start (default = false) -->\n" + \
				"        <recreateDBOnStart>true</recreateDBOnStart>\n" + \
				"\n" + \
				"        <!-- To apply the changes (default = false) -->\n" + \
				"        <applyCellChanges>true</applyCellChanges>\n" + \
				"\n" + \
				"        <!-- To  apply cell changes on a copy of the original target, with a custom suffix (default = " + \
				"true) -->\n" + \
				"        <cloneTargetSchema>true</cloneTargetSchema>\n" + \
				"        <cloneSuffix>_dirty</cloneSuffix>\n" + \
				"\n" + \
				"        <!-- To export the dirty db -->\n" + \
				"        <exportDirtyDB>true</exportDirtyDB>\n" + \
				"        <exportDirtyDBPath>/tmp</exportDirtyDBPath>\n" + \
				"        <exportDirtyDBType>CSV</exportDirtyDBType>\n" + \
				"\n" + \
				"        <!-- To export the changes -->\n" + \
				"        <exportCellChanges>false</exportCellChanges>\n" + \
				"\n" + \
				"        <!-- To compute an estimate of the reparability (default = false) -->\n" + \
				"        <estimateRepairability>false</estimateRepairability>\n" + \
				"\n" + \
				"        <!-- To generate all possible changes (default = false - slow, only for toy examples)-->\n" + \
				"        <generateAllChanges>false</generateAllChanges>\n" + \
				"\n" + \
				"        <!-- To avoid interactions among changes. (default = true) -->\n" + \
				"        <avoidInteractions>false</avoidInteractions>\n" + \
				"\n" + \
				"        <!-- To check, at the end of the process, if changes are detectable. (default = false) -->\n" + \
				"        <checkChanges>false</checkChanges>\n" + \
				"\n" + \
				"        <!-- To compute an estimate of the repairability. Requires checkChanges = true. (default = false)" + \
				" -->\n" + \
				"        <estimateRepairability>false</estimateRepairability>\n" + \
				"\n" + \
				"        <!-- To use an optimized strategy for updates. (default = true) -->\n" + \
				"        <useDeltaDBForChanges>true</useDeltaDBForChanges>\n"

		random_errors = "\n" + \
				"\t\t<randomErrors>\n" + \
				"            <tables>\n" + \
				"                <table name=\"person\"> \n" + \
				"                    <percentage>" + str(error_fractions[e] * 100)[0:5] + "</percentage> <!-- Percentage is wrt attributes to dirty in the table " + \
				"-->\n" + \
				"                    <attributes>\n" + \
				"                        <attribute>" + header[error_cols[e]] + "</attribute>\n" + \
				"                    </attributes>\n" + \
				"                </table>\n" + \
				"            </tables>\n" + \
				"        </randomErrors>\n" + \
				"\n" + \
				"        <!-- To control the way in which changing the value -->\n" + \
				"        <dirtyStrategies>\n" + \
				"            <defaultStrategy>\n" + \
				"                <strategy chars=\"*\" charsToAdd=\"1\">TypoAddString</strategy>\n" + \
				"            </defaultStrategy> \n" + \
				"            <attributeStrategy>\n" + \
				"                <attribute table=\"person\" name=\"" + header[error_cols[e]] + "\">\n"

		strat = ""
		if error_strategy[e] == 0:
			strat += "<strategy chars=\"*\" charsToAdd=\"1\">TypoAddString</strategy>\n"
		if error_strategy[e] == 1:
			strat += "<strategy charsToRemove=\"1\">TypoRemoveString</strategy>\n"
		if error_strategy[e] == 2:
			strat += "<strategy charsToSwitch=\"1\">TypoSwitchValue</strategy>\n"
		if error_strategy[e] == 3:
			strat += "<strategy>TypoActiveDomain</strategy>\n"
		if error_strategy[e] == 4:
			strat += "<strategy>TypoRandom</strategy>\n"


		end =	"                </attribute>\n" + \
				"            </attributeStrategy>\n" + \
				"        </dirtyStrategies>\n" + \
				"\n" + \
				"    </configuration>\n" + \
				"</task>"

		configuration = start + fd + conf + random_errors + strat + end

		file = open("/tmp/bart_config"+ str(e) +".xml", "w+")
		file.write(configuration)
		file.close()

		import subprocess
		subprocess.call('sh /home/felix/BART/Bart_Engine/run.sh /tmp/bart_config'+ str(e) +'.xml', shell=True)